"""命令行入口：交互模式 / 单次执行 / 内置自检。

用法示例::

    # 1) 离线演示（Mock 大模型 + Mock 传输层，不需要任何配置）
    python -m brain

    # 2) 内置自检：一次跑完正常/失败/超时/坏报文/越界重试等场景
    python -m brain --selftest

    # 3) 单次执行（脚本化 / CI 用）
    python -m brain --once "去迷宫右下角"

    # 4) 接真实大模型（通义千问 OpenAI 兼容模式）
    set BRAIN_LLM_API_KEY=sk-xxx
    python -m brain --llm openai

    # 5) TCP 联调彩排（先起模拟 C 端，再起大脑端）
    python tools/mock_c_server.py --port 9000
    python -m brain --transport tcp --tcp-port 9000
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .config import AppConfig
from .errors import BrainError, ConfigError, SessionBusyError, SessionClosedError
from .feedback import FeedbackRenderer
from .llm.base import LLMClient
from .llm.mock import MockLLM
from .llm.openai_compat import OpenAICompatConfig, OpenAICompatLLM
from .logging_setup import setup_logging
from .planner import TargetPlanner
from .protocol import parse_target_message
from .session import BrainSession, TaskStatus
from .transport.base import Transport
from .transport.mock import MockRobotTransport
from .transport.serial import SerialTransport
from .transport.tcp import TcpTransport

logger = logging.getLogger(__name__)

_QUIT_COMMANDS = ("q", "quit", "exit", "退出")
_HELP_COMMANDS = ("help", "帮助", "?")

_TASK_STATUS_LABELS = {
    TaskStatus.RUNNING: "寻路成功",
    TaskStatus.FAIL: "寻路失败",
    TaskStatus.INVALID_INSTRUCTION: "指令无法解析",
    TaskStatus.LLM_ERROR: "大模型调用失败",
    TaskStatus.TRANSPORT_ERROR: "通信异常",
    TaskStatus.PROTOCOL_ERROR: "回包协议异常",
    TaskStatus.TIMEOUT: "等待回包超时",
}


# --------------------------------------------------------------------------- #
# 组装组件（依赖注入）
# --------------------------------------------------------------------------- #
def build_llm(config: AppConfig) -> LLMClient:
    """根据配置创建大模型客户端。"""
    llm_config = config.llm
    if llm_config.provider == "mock":
        logger.info("使用离线 Mock 大模型（无需 API Key）")
        return MockLLM()
    if llm_config.provider == "openai":
        if not llm_config.api_key:
            raise ConfigError(
                "使用真实大模型需要配置 API Key：BRAIN_LLM_API_KEY"
                "（或 DASHSCOPE_API_KEY / OPENAI_API_KEY）"
            )
        logger.info("使用 OpenAI 兼容大模型：%s", llm_config.describe())
        return OpenAICompatLLM(
            OpenAICompatConfig(
                base_url=llm_config.base_url,
                model=llm_config.model,
                api_key=llm_config.api_key,
                timeout=llm_config.timeout,
                max_retries=llm_config.max_retries,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
            )
        )
    raise ConfigError(f"未知的大模型 provider：{llm_config.provider!r}")


def build_transport(config: AppConfig) -> Transport:
    """根据配置创建通信传输层。"""
    transport_config = config.transport
    if transport_config.kind == "mock":
        logger.info("使用 Mock 传输层（本地模拟 C 端）")
        return MockRobotTransport()
    if transport_config.kind == "tcp":
        logger.info(
            "使用 TCP 传输层：%s:%d", transport_config.host, transport_config.port
        )
        return TcpTransport(
            host=transport_config.host,
            port=transport_config.port,
            connect_timeout=transport_config.connect_timeout,
        )
    if transport_config.kind == "serial":
        logger.info(
            "使用串口传输层：%s @ %d bps",
            transport_config.serial_port,
            transport_config.baudrate,
        )
        return SerialTransport(
            port=transport_config.serial_port,
            baudrate=transport_config.baudrate,
        )
    raise ConfigError(f"未知的通信方式：{transport_config.kind!r}")


def build_session(config: AppConfig) -> BrainSession:
    """组装一次完整的“大脑会话”（大模型 + 规划器 + 传输层 + 状态机）。"""
    llm = build_llm(config)
    planner = TargetPlanner(llm, max_retries=config.llm.max_retries)
    transport = build_transport(config)
    return BrainSession(
        planner=planner,
        transport=transport,
        feedback=FeedbackRenderer(),
        report_timeout=config.report_timeout,
    )


# --------------------------------------------------------------------------- #
# 交互模式 / 单次执行
# --------------------------------------------------------------------------- #
def _print_banner(config: AppConfig) -> None:
    print("=" * 72)
    print(" 具身 Agent 小车 —— Python 大脑端（自然语言 -> 目标坐标 -> C 端）")
    print("=" * 72)
    print(f" 大模型：{config.llm.describe()}")
    transport_desc = {
        "mock": "Mock 传输层（本地模拟 C 端，无需硬件）",
        "tcp": f"TCP -> {config.transport.host}:{config.transport.port}",
        "serial": (
            f"串口 {config.transport.serial_port} @ {config.transport.baudrate} bps"
        ),
    }.get(config.transport.kind, config.transport.kind)
    print(f" 通信  ：{transport_desc}")
    print(f" 坐标  ：6×6 栅格，左上角 (0,0)，x 向下、y 向右，范围 0~5")
    print(f" 提示  ：输入 help 查看帮助，输入 q 退出")
    print("-" * 72)


def run_interactive(config: AppConfig) -> int:
    """交互模式：连续对话，跑完一个任务可继续下一条指令。"""
    feedback = FeedbackRenderer()
    try:
        setup_logging(config.verbose, config.log_file)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    try:
        session = build_session(config)
    except (ConfigError, ValueError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 2

    _print_banner(config)
    try:
        session.open()
    except BrainError as exc:
        print(f"连接失败：{exc.message}", file=sys.stderr)
        return 2

    try:
        while True:
            try:
                text = input("请输入指令 > ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            text = text.strip()
            if not text:
                print("（空输入：请输入指令，或输入 help 查看帮助）")
                continue
            if text.lower() in _QUIT_COMMANDS:
                break
            if text in _HELP_COMMANDS:
                print(feedback.help_text())
                continue

            try:
                outcome = session.handle_instruction(text)
            except SessionBusyError as exc:
                print(f"[大脑] {exc.message}")
                continue
            except SessionClosedError as exc:  # 理论上不会发生，防御式处理
                print(f"[大脑] {exc.message}")
                break
            except KeyboardInterrupt:
                print()
                break
            except BrainError as exc:
                print(f"[大脑] 出错了：{exc.message}")
                continue
            except Exception:  # noqa: BLE001 - 第三方插件异常兜底，避免整个主循环崩溃
                logger.exception("处理指令时发生未预期异常")
                print("[大脑] 内部错误，已记录详细日志；可继续输入下一条指令。")
                continue

            print()
            print(f"[大脑] {outcome.user_message}")
            print()
    finally:
        session.close()

    print(feedback.goodbye())
    return 0


def run_once(config: AppConfig, instruction: str) -> int:
    """单次执行一条指令（脚本化 / CI 用）。"""
    try:
        setup_logging(config.verbose, config.log_file)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    feedback = FeedbackRenderer()
    try:
        session = build_session(config)
    except (ConfigError, ValueError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 2

    try:
        session.open()
        outcome = session.handle_instruction(instruction)
    except BrainError as exc:
        print(f"执行失败：{exc.message}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001 - 未预期异常兜底，避免直接抛 traceback
        logger.exception("执行指令时发生未预期异常")
        print("执行失败：内部错误，已记录详细日志。", file=sys.stderr)
        return 2
    finally:
        session.close()

    print(outcome.user_message)
    logger.info("任务状态：%s", _TASK_STATUS_LABELS.get(outcome.status, outcome.status.value))
    return 0 if outcome.is_ok else 1


# --------------------------------------------------------------------------- #
# 内置自检（不需要任何配置 / 硬件 / 网络）
# --------------------------------------------------------------------------- #
def _run_selftest_case(
    name: str,
    instruction: str,
    llm: LLMClient,
    transport: Transport,
    expected_status: TaskStatus,
    expected_keywords: tuple[str, ...],
) -> bool:
    feedback = FeedbackRenderer()
    planner = TargetPlanner(llm, max_retries=2)
    session = BrainSession(
        planner=planner, transport=transport, feedback=feedback, report_timeout=1.0
    )

    print(f"\n[自检] {name}")
    print(f"  用户指令：{instruction}")
    session.open()
    try:
        outcome = session.handle_instruction(instruction)
    finally:
        session.close()

    calls = getattr(llm, "calls", [])
    sent_lines = getattr(transport, "sent_lines", [])
    print(f"  大模型调用次数：{len(calls)}")
    print(f"  下发报文：{[line.strip() for line in sent_lines]}")
    print(f"  回包：{outcome.raw_report!r}")
    print(f"  反馈：{outcome.user_message}")
    print(f"  任务状态：{_TASK_STATUS_LABELS.get(outcome.status, outcome.status.value)}")

    ok = outcome.status == expected_status
    ok = ok and all(keyword in outcome.user_message for keyword in expected_keywords)
    if expected_status is TaskStatus.INVALID_INSTRUCTION:
        ok = ok and not sent_lines  # 不合法的目标绝不能下发
    if ok and outcome.target is not None and sent_lines:
        try:
            ok = parse_target_message(sent_lines[-1]) == outcome.target
        except Exception:  # noqa: BLE001 - 自检里只关心是否通过
            ok = False
    print("  结果：PASS" if ok else "  结果：FAIL")
    if not ok:
        print(f"  （期望状态：{expected_status.value}；期望关键词：{expected_keywords}）")
    return ok


def run_selftest() -> int:
    """本地自检：覆盖第 1、2 步（基础 demo + 回包解析/异常处理）。"""
    setup_logging(verbose=False, log_file=None)
    # 自检时只保留自检脚本的整洁输出，日志降级（自检本身会打印完整对话记录）
    logging.getLogger("brain").setLevel(logging.CRITICAL)

    cases: list[tuple[str, str, LLMClient, Transport, TaskStatus, tuple[str, ...]]] = [
        (
            "正常寻路（MockLLM + Mock 传输层）",
            "去迷宫右下角",
            MockLLM(),
            MockRobotTransport(),
            TaskStatus.RUNNING,
            ("已规划路线",),
        ),
        (
            "障碍物堵死（目标点被占据）",
            "去(1,1)",
            MockLLM(),
            MockRobotTransport(),
            TaskStatus.FAIL,
            ("阻挡",),
        ),
        (
            "坐标越界后自动重试成功",
            "去(9,9)",
            MockLLM(
                scripted_outputs=[
                    '{"target_x": 9, "target_y": 9}',
                    '{"target_x": 3, "target_y": 2}',
                ]
            ),
            MockRobotTransport(),
            TaskStatus.RUNNING,
            ("已规划路线",),
        ),
        (
            "模型输出始终不合法（重试用尽）",
            "去(3,2)",
            MockLLM(scripted_outputs=["这是一段乱码"] * 3),
            MockRobotTransport(),
            TaskStatus.INVALID_INSTRUCTION,
            ("无法理解",),
        ),
        (
            "模型无法确定目标（-1,-1）",
            "随便走走吧",
            MockLLM(),
            MockRobotTransport(),
            TaskStatus.INVALID_INSTRUCTION,
            ("无法确定目标",),
        ),
        (
            "等待回包超时",
            "回到(0,0)起点",
            MockLLM(),
            MockRobotTransport(scenario="timeout"),
            TaskStatus.TIMEOUT,
            ("响应",),
        ),
        (
            "回包 JSON 非法",
            "去(3,2)",
            MockLLM(),
            MockRobotTransport(scenario="bad_json"),
            TaskStatus.PROTOCOL_ERROR,
            ("格式异常",),
        ),
        (
            "回包未知状态",
            "去(3,2)",
            MockLLM(),
            MockRobotTransport(scenario="unknown_status"),
            TaskStatus.PROTOCOL_ERROR,
            ("未知状态",),
        ),
        (
            "通信发送异常",
            "去(2,2)",
            MockLLM(),
            MockRobotTransport(scenario="send_error"),
            TaskStatus.TRANSPORT_ERROR,
            ("通信异常",),
        ),
    ]

    print("=" * 72)
    print(" 内置自检：基础 demo + 回包解析 + 异常处理（不需要网络 / 硬件）")
    print("=" * 72)
    passed = 0
    for case in cases:
        if _run_selftest_case(*case):
            passed += 1
    print("\n" + "=" * 72)
    print(f" 自检结果：{passed}/{len(cases)} 通过")
    print("=" * 72)
    return 0 if passed == len(cases) else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m brain",
        description="基于大语言模型的自然语言指令式视觉避障机器人 —— Python 大脑端",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="运行内置自检（离线，不需要配置/硬件）",
    )
    parser.add_argument(
        "--once",
        metavar="指令",
        help="只执行一条指令后退出（脚本化/CI 用），例如：--once \"去右下角\"",
    )
    parser.add_argument(
        "--llm",
        choices=("mock", "openai"),
        help="大模型 provider（默认读环境变量；无 Key 时为 mock）",
    )
    parser.add_argument("--model", help="模型名，例如 qwen-plus / qwen2.5:7b")
    parser.add_argument("--base-url", help="OpenAI 兼容接口 base_url")
    parser.add_argument("--api-key", help="API Key（也可用环境变量配置）")
    parser.add_argument("--llm-timeout", type=float, help="大模型请求超时（秒）")
    parser.add_argument(
        "--transport",
        choices=("mock", "tcp", "serial"),
        help="通信方式（默认 mock）",
    )
    parser.add_argument("--tcp-host", help="TCP 主机")
    parser.add_argument("--tcp-port", type=int, help="TCP 端口")
    parser.add_argument("--serial-port", help="串口号，例如 COM3")
    parser.add_argument("--baudrate", type=int, help="串口波特率，默认 115200")
    parser.add_argument(
        "--report-timeout",
        type=float,
        help="等待 C 端回包超时（秒），默认 5",
    )
    parser.add_argument("--log-file", help="日志文件路径（默认只输出到控制台）")
    parser.add_argument("--env-file", help="环境变量文件路径（默认项目根目录 .env）")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.llm is not None:
        config = config.with_llm(provider=args.llm)
    if args.model is not None:
        config = config.with_llm(model=args.model)
    if args.base_url is not None:
        config = config.with_llm(base_url=args.base_url)
    if args.api_key is not None:
        config = config.with_llm(api_key=args.api_key)
    if args.llm_timeout is not None:
        config = config.with_llm(timeout=args.llm_timeout)
    if args.transport is not None:
        config = config.with_transport(kind=args.transport)
    if args.tcp_host is not None:
        config = config.with_transport(host=args.tcp_host)
    if args.tcp_port is not None:
        config = config.with_transport(port=args.tcp_port)
    if args.serial_port is not None:
        config = config.with_transport(serial_port=args.serial_port)
    if args.baudrate is not None:
        config = config.with_transport(baudrate=args.baudrate)
    if args.report_timeout is not None:
        config = config.with_app(report_timeout=args.report_timeout)
    if args.log_file is not None:
        config = config.with_app(log_file=args.log_file)
    if args.verbose:
        config = config.with_app(verbose=True)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()

    try:
        config = AppConfig.from_env(env_file=args.env_file)
        config = _apply_cli_overrides(config, args)
    except (ConfigError, ValueError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    if args.once is not None:
        return run_once(config, args.once)
    return run_interactive(config)