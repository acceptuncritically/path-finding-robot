"""大模型原始输出的解析：把“模型说的话”提取成合法的目标坐标。

真实大模型的输出经常“不听话”：
- 带 Markdown 代码围栏；
- 前后带解释文字；
- 坐标是字符串 ``"5"``、浮点 ``5.0``、布尔 ``true``；
- 超出 6×6 范围。

**容错边界（安全优先）**
1. 只认包含 ``target_x`` / ``target_y`` 的 JSON 对象；出现**多个不同目标**时
   不猜测，直接报错让上层重试；
2. 文本明确说“无法确定”且没有合法目标 JSON 时，不猜坐标，交给上层问用户；
3. 正则兜底**只对“整段近似纯字段表达式”生效**（例如 ``target_x: 3, target_y: 2``），
   避免把模型解释文字里的“示例坐标”当成真实目标下发；
4. 只有**整数** ``-1, -1`` 才视为“无法确定”约定标记；``-1.0`` 属于浮点，按格式错误处理。
"""

from __future__ import annotations

import json
import re

from ..errors import TargetFormatError, TargetOutOfRangeError, TargetUncertainError
from ..protocol import MAX_COORD, MIN_COORD, Target

# ```json ... ``` 或 ``` ... ```
_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# 文本中的 {...}（非贪婪，逐个提取）
_OBJECT_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)
# 兜底：target_x: 3 / "target_x": 3
_FIELD_PATTERN = re.compile(r'"?target_([xy])"?\s*[:：]\s*(-?\d+)', re.IGNORECASE)
# 模型明确表示“无法确定”的措辞
_UNCERTAIN_PATTERN = re.compile(
    r"无法确定|不能确定|不确定|不知道|无法识别|无法判断|没有找到"
)
# 纯字段表达式允许出现的剩余字符（去掉字段后应只剩这些）
_BARE_ALLOWED_PATTERN = re.compile(r"[\s,，;；:：{}\[\]()（）\"'`。.!！?？、]")


def parse_target_from_llm_output(raw: str) -> Target:
    """解析大模型原始输出，返回合法 ``Target``。

    Raises:
        TargetFormatError: 无法解析出唯一目标 JSON / 字段缺失 / 类型错误 /
            输出中出现多个不同目标。
        TargetOutOfRangeError: 坐标超出 0~5 范围。
        TargetUncertainError: 模型表示无法确定目标坐标。
    """
    if not isinstance(raw, str):
        raise TargetFormatError(f"大模型输出必须是字符串，实际为 {type(raw).__name__}", "")
    text = raw.strip()
    if not text:
        raise TargetFormatError("大模型输出为空", raw)

    targets: list[Target] = []
    errors: list[Exception] = []
    for candidate in _candidate_json_texts(text):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, RecursionError):
            continue
        if not isinstance(data, dict):
            continue
        if "target_x" not in data and "target_y" not in data:
            continue
        try:
            target = _validate_target_values(data, raw)
        except (TargetFormatError, TargetOutOfRangeError, TargetUncertainError) as exc:
            errors.append(exc)
            continue
        if target not in targets:
            targets.append(target)

    if len(targets) > 1:
        raise TargetFormatError(
            "输出中出现多个不同的目标坐标，无法确定以哪个为准（请只输出一个 JSON）", raw
        )
    if len(targets) == 1:
        return targets[0]

    if errors:
        # 都是非法目标时，优先上报更具体的“越界”错误；否则报第一个错误
        for error in errors:
            if isinstance(error, TargetOutOfRangeError):
                raise error
        first = errors[0]
        if isinstance(first, TargetUncertainError):
            raise first
        raise TargetFormatError(first.message, raw)  # type: ignore[attr-defined]

    # 没有可解析的目标 JSON：若模型明确说“无法确定”，不猜坐标
    if _UNCERTAIN_PATTERN.search(text):
        raise TargetUncertainError("模型表示无法确定目标坐标", raw)

    # 最后兜底：仅当整段文本近似“纯字段表达式”时才接受
    fallback = _try_parse_fields(text, raw)
    if fallback is not None:
        return fallback

    raise TargetFormatError(
        "输出中找不到合法的目标坐标 JSON（应形如 "
        '{"target_x": 3, "target_y": 2}）',
        raw,
    )


def _candidate_json_texts(text: str) -> list[str]:
    """按优先级生成待尝试的 JSON 文本片段（围栏 > 整段 > 各 {...}）。"""
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        stripped = candidate.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            candidates.append(stripped)

    for match in _FENCE_PATTERN.finditer(text):
        _add(match.group(1))
    _add(text)
    for match in _OBJECT_PATTERN.finditer(text):
        _add(match.group(0))
    return candidates


def _validate_target_values(data: dict[str, object], raw: str) -> Target:
    x = data.get("target_x")
    y = data.get("target_y")
    if "target_x" not in data or "target_y" not in data:
        raise TargetFormatError(f"目标 JSON 缺少字段：{sorted(data) or '空对象'}", raw)
    if type(x) is not int or type(y) is not int:
        raise TargetFormatError(
            f"target_x/target_y 必须是 JSON 整数，实际为 {x!r}/{y!r}", raw
        )
    if x == -1 and y == -1:
        raise TargetUncertainError(
            "模型表示无法确定目标坐标（返回 -1,-1 约定标记）", raw
        )
    if not (MIN_COORD <= x <= MAX_COORD and MIN_COORD <= y <= MAX_COORD):
        raise TargetOutOfRangeError(
            f"坐标超出 {MIN_COORD}~{MAX_COORD} 范围：x={x}, y={y}", raw
        )
    return Target(x=x, y=y)


def _try_parse_fields(text: str, raw: str) -> Target | None:
    """纯字段表达式兜底，例如 ``target_x: 3, target_y: 2``。

    只有去掉字段与标点后**没有任何剩余文字**时才接受，避免把解释/示例里的
    坐标误当成真实目标。
    """
    if not _FIELD_PATTERN.search(text):
        return None

    remainder = _FIELD_PATTERN.sub("", text)
    remainder = _BARE_ALLOWED_PATTERN.sub("", remainder)
    if remainder:
        return None

    found: dict[str, int] = {}
    for match in _FIELD_PATTERN.finditer(text):
        found[match.group(1).lower()] = int(match.group(2))
    if "x" not in found or "y" not in found:
        return None

    x, y = found["x"], found["y"]
    if x == -1 and y == -1:
        raise TargetUncertainError(
            "模型表示无法确定目标坐标（返回 -1,-1 约定标记）", raw
        )
    if not (MIN_COORD <= x <= MAX_COORD and MIN_COORD <= y <= MAX_COORD):
        raise TargetOutOfRangeError(
            f"坐标超出 {MIN_COORD}~{MAX_COORD} 范围：x={x}, y={y}", raw
        )
    return Target(x=x, y=y)