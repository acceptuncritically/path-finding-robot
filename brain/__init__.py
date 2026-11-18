"""基于大语言模型的自然语言指令式视觉避障机器人 —— Python 大脑端。

分层结构（由内到外，均通过接口解耦）：

- ``brain.protocol``  与 C 端（机器人躯体底层）之间的固定 JSON 协议
- ``brain.grid``      6×6 栅格工具（BFS 路径，供模拟 C 端使用）
- ``brain.prompt``    大模型系统提示词
- ``brain.llm``       大模型客户端抽象（Mock / OpenAI 兼容）
- ``brain.planner``   自然语言 -> 目标坐标（调用大模型 + 解析 + 重试 + 校验）
- ``brain.transport`` 传输层抽象（Mock / TCP / 串口）
- ``brain.feedback``  人话反馈渲染
- ``brain.session``   任务状态机（与输入输出方式无关，可被控制台 / Web 复用）
- ``brain.cli``       命令行入口
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]