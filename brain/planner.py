"""目标规划器：自然语言 -> 目标栅格坐标。

职责边界（很重要，这是解耦的关键）：
- 只依赖 ``LLMClient`` 抽象，不关心模型是 Mock、通义千问还是 LLaMA；
- 只做“调用 -> 解析 -> 校验 -> 失败重试 -> 最终失败归类”；
- 不碰通信层、不打印给用户的话（人话反馈由 ``brain.feedback`` 负责）。

失败重试策略
------------
- 输出格式错误 / 坐标越界：把错误原因回喂给模型，最多重试 ``max_retries`` 次；
- 模型明确返回 -1,-1（无法确定目标）：直接失败，提示用户补充信息（重试无意义）；
- 大模型调用失败（网络/鉴权）：交给 LLM 客户端内部重试，仍失败则直接失败。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .errors import (
    LLMError,
    LLMOutputError,
    PlanningFailure,
    TargetOutOfRangeError,
    TargetUncertainError,
)
from .llm.base import LLMClient
from .llm.parser import parse_target_from_llm_output
from .prompt import SYSTEM_PROMPT, build_retry_message, build_user_message
from .protocol import Target

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanResult:
    """一次成功规划的结果。"""

    target: Target
    attempts: int
    raw_outputs: tuple[str, ...]


class TargetPlanner:
    """把用户指令规划成 6×6 栅格上的目标坐标。"""

    def __init__(self, llm: LLMClient, max_retries: int = 2) -> None:
        if max_retries < 0:
            raise ValueError(f"max_retries 不能为负数：{max_retries}")
        self._llm = llm
        self._max_retries = max_retries

    @property
    def llm(self) -> LLMClient:
        return self._llm

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def plan(self, instruction: str) -> PlanResult:
        """解析一条自然语言指令并返回目标坐标。

        Raises:
            PlanningFailure: 最终失败（已用尽重试），带失败类别与最后一次原始输出。
        """
        text = (instruction or "").strip()
        if not text:
            raise PlanningFailure("指令为空", kind="format", attempts=0)

        raw_outputs: list[str] = []
        last_error: LLMOutputError | None = None
        user_message = build_user_message(text)
        total_attempts = self._max_retries + 1

        for attempt in range(1, total_attempts + 1):
            logger.info(
                "第 %d/%d 次调用大模型，用户指令：%s", attempt, total_attempts, text
            )
            try:
                raw = self._llm.complete(SYSTEM_PROMPT, user_message)
            except LLMError as exc:
                logger.error("大模型调用失败：%s", exc)
                raise PlanningFailure(
                    str(exc),
                    kind="llm",
                    attempts=attempt,
                    last_raw=raw_outputs[-1] if raw_outputs else "",
                ) from exc

            raw_outputs.append(raw)
            logger.debug("大模型原始输出（第 %d 次）：%r", attempt, raw)

            try:
                target = parse_target_from_llm_output(raw)
            except TargetUncertainError as exc:
                logger.warning("模型无法确定目标：%s", exc.message)
                raise PlanningFailure(
                    exc.message,
                    kind="uncertain",
                    attempts=attempt,
                    last_raw=raw,
                ) from exc
            except LLMOutputError as exc:
                last_error = exc
                logger.warning("大模型输出不可用（第 %d 次）：%s", attempt, exc.message)
                if attempt < total_attempts:
                    user_message = build_retry_message(text, exc.message, raw)
                continue

            logger.info("目标解析成功：%s（第 %d 次尝试）", target, attempt)
            return PlanResult(
                target=target,
                attempts=attempt,
                raw_outputs=tuple(raw_outputs),
            )

        kind = "range" if isinstance(last_error, TargetOutOfRangeError) else "format"
        message = last_error.message if last_error else "大模型输出解析失败"
        raise PlanningFailure(
            message,
            kind=kind,
            attempts=total_attempts,
            last_raw=raw_outputs[-1] if raw_outputs else "",
        )