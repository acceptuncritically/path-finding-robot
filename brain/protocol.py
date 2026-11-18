"""与 C 端（机器人躯体底层）之间的固定通信协议 v1.0（字段冻结）。

协议原文
--------
- Python -> C（下发指令）::

      {"target_x": 5, "target_y": 5}

- C -> Python（寻路成功）::

      {"status":"running","path":[{"x":0,"y":0},{"x":1,"y":0}]}

- C -> Python（寻路失败）::

      {"status":"fail","path":[]}

坐标系
------
6×6 栅格，左上角为 (0,0)，x 向下（行号），y 向右（列号），取值 0~5 的整数。

本模块只负责“报文的组装 / 解析 / 校验”，不做任何 IO，便于单元测试与复用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from .errors import (
    InvalidReportPathError,
    InvalidTargetMessageError,
    MalformedReportError,
    UnknownStatusError,
)

# --------------------------------------------------------------------------- #
# 常量（协议的一部分）
# --------------------------------------------------------------------------- #
GRID_SIZE = 6
MIN_COORD = 0
MAX_COORD = GRID_SIZE - 1
ALLOWED_STATUSES = ("running", "fail")
LINE_END = "\n"


# --------------------------------------------------------------------------- #
# 值对象
# --------------------------------------------------------------------------- #
def _ensure_coord(value: object, field_name: str) -> int:
    """校验单个坐标分量：必须是 0~5 的整数（bool 不是合法整数）。"""
    if type(value) is not int:
        raise ValueError(
            f"{field_name} 必须是整数，实际为 {type(value).__name__}: {value!r}"
        )
    if not (MIN_COORD <= value <= MAX_COORD):
        raise ValueError(f"{field_name} 超出 {MIN_COORD}~{MAX_COORD} 范围：{value}")
    return value


@dataclass(frozen=True)
class Target:
    """目标栅格坐标（下发指令里的 target_x / target_y）。"""

    x: int
    y: int

    def __post_init__(self) -> None:
        _ensure_coord(self.x, "target_x")
        _ensure_coord(self.y, "target_y")

    def as_dict(self) -> dict[str, int]:
        return {"target_x": self.x, "target_y": self.y}

    def human(self) -> str:
        """人话描述，例如 ``(x=5, y=5)（第 6 行第 6 列）``。"""
        return f"(x={self.x}, y={self.y})（第 {self.x + 1} 行第 {self.y + 1} 列）"

    def __str__(self) -> str:
        return f"({self.x},{self.y})"


@dataclass(frozen=True)
class Point:
    """路径上的一个栅格点（C 端上报的 path[i]）。"""

    x: int
    y: int

    def __post_init__(self) -> None:
        _ensure_coord(self.x, "x")
        _ensure_coord(self.y, "y")

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}

    def __str__(self) -> str:
        return f"({self.x},{self.y})"


@dataclass(frozen=True)
class RobotReport:
    """C 端上报的机器人状态/路径（解析后的内部结构）。"""

    status: str
    path: tuple[Point, ...] = ()
    raw: str = ""

    @property
    def is_success(self) -> bool:
        return self.status == "running"

    @property
    def step_count(self) -> int:
        """从起点走到终点的步数（路径点数 - 1；单点路径为 0 步）。"""
        return max(len(self.path) - 1, 0)

    def path_text(self) -> str:
        return format_path(self.path)


# --------------------------------------------------------------------------- #
# Python -> C：下发指令
# --------------------------------------------------------------------------- #
def build_target_message(target: Target) -> str:
    """严格按协议组装下发报文：紧凑单行 JSON + ``\\n`` 帧边界。"""
    if not isinstance(target, Target):
        raise TypeError(
            f"build_target_message 需要 Target 对象，实际为 {type(target).__name__}"
        )
    payload = {"target_x": target.x, "target_y": target.y}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + LINE_END


def parse_target_message(raw: str) -> Target:
    """解析一条下发报文（C 端模拟器 / 单元测试使用）。

    Raises:
        InvalidTargetMessageError: 报文不是合法 JSON、缺字段、类型错误或越界。
    """
    text = _clean_line(raw)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, RecursionError) as exc:
        reason = exc.msg if isinstance(exc, json.JSONDecodeError) else "JSON 嵌套过深"
        raise InvalidTargetMessageError(f"下发报文不是合法 JSON：{reason}", text) from exc
    if not isinstance(data, dict):
        raise InvalidTargetMessageError("下发报文必须是 JSON 对象", text)

    for field_name in ("target_x", "target_y"):
        if field_name not in data:
            raise InvalidTargetMessageError(f"下发报文缺少 {field_name} 字段", text)
        value = data[field_name]
        if type(value) is not int:
            raise InvalidTargetMessageError(
                f"{field_name} 必须是 JSON 整数，实际为 {value!r}", text
            )
        if not (MIN_COORD <= value <= MAX_COORD):
            raise InvalidTargetMessageError(
                f"{field_name} 超出 {MIN_COORD}~{MAX_COORD} 范围：{value}", text
            )
    return Target(x=data["target_x"], y=data["target_y"])


# --------------------------------------------------------------------------- #
# C -> Python：上报状态/路径
# --------------------------------------------------------------------------- #
def parse_robot_report(raw: str) -> RobotReport:
    """解析 C 端一行 JSON 回包，严格校验结构、类型、范围与 status 语义。

    Raises:
        MalformedReportError: 非 JSON / 顶层不是对象 / 缺 status 或 path 字段。
        UnknownStatusError: status 不在 ``{"running", "fail"}`` 内。
        InvalidReportPathError: path 结构或坐标非法，或与 status 语义不匹配。
    """
    text = _clean_line(raw)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, RecursionError) as exc:
        reason = exc.msg if isinstance(exc, json.JSONDecodeError) else "JSON 嵌套过深"
        raise MalformedReportError(f"不是合法 JSON：{reason}", text) from exc
    if not isinstance(data, dict):
        raise MalformedReportError("回包必须是 JSON 对象", text)

    if "status" not in data:
        raise MalformedReportError("回包缺少 status 字段", text)
    status = data["status"]
    if not isinstance(status, str) or status not in ALLOWED_STATUSES:
        raise UnknownStatusError(status, text)

    if "path" not in data:
        raise MalformedReportError("回包缺少 path 字段", text)
    raw_path = data["path"]
    if not isinstance(raw_path, list):
        raise InvalidReportPathError("path 必须是数组", text)

    points: list[Point] = []
    for index, item in enumerate(raw_path):
        if not isinstance(item, dict):
            raise InvalidReportPathError(f"path[{index}] 必须是坐标对象", text)
        if "x" not in item or "y" not in item:
            raise InvalidReportPathError(f"path[{index}] 缺少 x/y 字段", text)
        x, y = item["x"], item["y"]
        if type(x) is not int or type(y) is not int:
            raise InvalidReportPathError(
                f"path[{index}] 的 x/y 必须是 JSON 整数，实际为 {x!r}/{y!r}", text
            )
        if not (MIN_COORD <= x <= MAX_COORD and MIN_COORD <= y <= MAX_COORD):
            raise InvalidReportPathError(f"path[{index}] 坐标越界：({x},{y})", text)
        points.append(Point(x=x, y=y))

    if status == "fail" and points:
        raise InvalidReportPathError("status=fail 但 path 非空（协议要求 path 为 []）", text)
    if status == "running" and not points:
        raise InvalidReportPathError(
            "status=running 但 path 为空（协议歧义，需与 C 端确认路径语义）", text
        )

    return RobotReport(status=status, path=tuple(points), raw=text)


# --------------------------------------------------------------------------- #
# 辅助函数（模拟 C 端 / 测试 / 调试用）
# --------------------------------------------------------------------------- #
def build_robot_report(
    status: str,
    points: Iterable[Point | tuple[int, int]] = (),
) -> str:
    """组装一条 C 端风格的 JSON 回包（紧凑单行，不含末尾换行）。"""
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"status 必须是 {ALLOWED_STATUSES} 之一：{status!r}")
    normalized: list[Point] = []
    for item in points:
        if isinstance(item, Point):
            normalized.append(item)
        else:
            x, y = item
            normalized.append(Point(x=x, y=y))
    if status == "fail" and normalized:
        raise ValueError("status=fail 时 path 必须为空")
    if status == "running" and not normalized:
        raise ValueError("status=running 时 path 不能为空")
    payload = {
        "status": status,
        "path": [point.as_dict() for point in normalized],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_running_report(points: Iterable[Point | tuple[int, int]]) -> str:
    return build_robot_report("running", points)


def build_fail_report() -> str:
    return build_robot_report("fail", ())


def format_path(path: Iterable[Point]) -> str:
    """把路径格式化为 ``[(0,0),(1,0),...]``。"""
    return "[" + ",".join(f"({point.x},{point.y})" for point in path) + "]"


def _clean_line(raw: str) -> str:
    if not isinstance(raw, str):
        raise TypeError(f"报文必须是 str，实际为 {type(raw).__name__}")
    return raw.strip()