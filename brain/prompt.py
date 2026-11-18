"""大模型提示词（Prompt）。

提示词是“自然语言 -> 目标坐标”效果的关键，单独成模块便于：
- 迭代优化（改提示词不需要动业务代码）；
- 单元测试（校验提示词包含坐标约定与输出约束）；
- 未来做多轮上下文 / 指令模板时复用。
"""

from __future__ import annotations

# 系统提示词：约束模型只输出一行严格 JSON，坐标遵循 6×6 栅格约定
SYSTEM_PROMPT = """你是一个 6×6 迷宫小车的大脑模块，负责把用户的中文自然语言指令转换成目标栅格坐标。

【坐标约定】
- 迷宫为 6×6 栅格，坐标范围 0~5；
- 左上角为 (0,0)；
- x 轴向下（行号，越往下越大），y 轴向右（列号，越往右越大）；
- 右下角为 (5,5)。

【输出要求】（必须严格遵守，违反即失败）
- 只输出一行严格 JSON 对象，格式：{"target_x": 整数, "target_y": 整数}
- 只能包含 target_x、target_y 两个字段；
- 两个值都必须是 0~5 的 JSON 整数（不要加引号、不要小数、不要 null）；
- 不要输出解释、单位、Markdown 代码块、注释或任何多余文字。

【语义映射示例】
- "小车走到坐标(5,5)" → {"target_x":5,"target_y":5}
- "去迷宫右下角" → {"target_x":5,"target_y":5}
- "回到(0,0)起点" → {"target_x":0,"target_y":0}
- "起点对面" → {"target_x":5,"target_y":5}
- "去中间区域" → {"target_x":2,"target_y":2}
- 用户使用相对指令（如"往前走两格""右边一格"）时，若能从上下文推断绝对目标则输出绝对坐标；
- 如果指令完全无法确定目标（例如"随便走走吧"），输出 {"target_x":-1,"target_y":-1}。"""


def build_user_message(instruction: str) -> str:
    """把用户指令包装成第一轮 user 消息。"""
    return f"用户指令：{instruction.strip()}"


def build_retry_message(
    instruction: str,
    error_reason: str,
    previous_output: str,
) -> str:
    """解析失败后的纠正消息（附上一次输出与错误原因，要求模型只输出 JSON）。"""
    return (
        f"用户指令：{instruction.strip()}\n\n"
        f"你上一次的输出不可用（原因：{error_reason}）。\n"
        f"上一次输出：{previous_output!r}\n"
        '请严格只输出一行 JSON，例如：{"target_x": 3, "target_y": 2}\n'
        "要求：target_x、target_y 均为 0~5 的整数，除 JSON 外不要输出任何字符。"
    )