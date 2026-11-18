"""人话反馈：把内部结果/异常翻译成面向用户的中文。

单独成模块的原因：
- 文案可以独立迭代，不影响业务逻辑；
- 将来做多语言、Web 前端渲染时，只需替换本类的实现（或子类化）。
"""

from __future__ import annotations

from .errors import PlanningFailure, ProtocolError, TransportError, UnknownStatusError
from .protocol import RobotReport, Target


class FeedbackRenderer:
    """控制台中文反馈渲染器。"""

    # ------------------------------------------------------------------ #
    # 成功 / 失败（C 端回包）
    # ------------------------------------------------------------------ #
    def running(self, report: RobotReport, target: Target | None = None) -> str:
        """C 端返回 status=running 且已规划出路径。"""
        if report.step_count == 0 and len(report.path) == 1:
            point = report.path[0]
            return f"小车当前已在目标点 ({point.x},{point.y})，无需移动。"
        target_text = f"目标点 {target.human()}；" if target is not None else ""
        return (
            f"目标解析完成，{target_text}已规划路线，小车准备出发，"
            f"路径：{report.path_text()}（共 {report.step_count} 步）"
        )

    def planning_failure(self, failure: PlanningFailure) -> str:
        """自然语言 -> 坐标这一步最终失败。"""
        if failure.kind == "uncertain":
            return (
                "无法确定目标位置，请给出具体坐标（如：去(3,2)）"
                "或方位（如：右下角、起点对面）。"
            )
        if failure.kind == "range":
            return f"坐标超出迷宫范围（有效范围 0~5）：{failure.message}。请重新输入。"
        if failure.kind == "llm":
            return f"大模型调用失败：{failure.message}。请检查网络或 API 配置后重试。"
        return (
            f"无法理解这条指令（{failure.message}），"
            "请直接给出坐标或方位，例如：去(3,2)、右下角。"
        )

    def fail(self, target: Target | None = None) -> str:
        """C 端返回 status=fail：障碍物堵死，无可行路径。"""
        return "目标点被障碍物阻挡，无法规划可行路线，请更换目标。"

    # ------------------------------------------------------------------ #
    # 异常
    # ------------------------------------------------------------------ #
    def protocol_error(self, exc: ProtocolError) -> str:
        """C 端回包不符合协议。"""
        if isinstance(exc, UnknownStatusError):
            return f"小车回传了未知状态：{exc.status!r}，请检查小车程序版本。"
        return (
            f"小车回传数据格式异常（原因：{exc.reason}），"
            "已记录原始报文，请检查小车程序。"
        )

    def transport_error(self, exc: TransportError) -> str:
        """TCP / 串口通信异常。"""
        return f"串口/网络通信异常：{exc.message}。请检查连接后重试。"

    def timeout(self, seconds: float) -> str:
        """等待 C 端回包超时。"""
        return (
            f"小车未在 {seconds:g} 秒内响应，"
            "请检查串口/网络连接后重试。"
        )

    # ------------------------------------------------------------------ #
    # 命令行交互
    # ------------------------------------------------------------------ #
    def goodbye(self) -> str:
        return "已退出，再见。"

    def help_text(self) -> str:
        return (
            "可用指令：\n"
            "  q / quit / exit / 退出   退出程序\n"
            "  help / 帮助              查看本帮助\n"
            "  任意中文指令             例如：\n"
            "    - 去迷宫右下角\n"
            "    - 小车走到坐标(5,5)\n"
            "    - 回到(0,0)起点\n"
            "\n"
            "坐标约定：6×6 栅格，左上角 (0,0)，x 向下（行）、y 向右（列），范围 0~5。"
        )