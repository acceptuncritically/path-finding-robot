"""模拟 C 端（机器人躯体底层）的 TCP 服务器：没有硬件时用来彩排联调。

用途
----
1. 大脑端的 TCP 传输层还没对接真实 C 程序时，先用本脚本模拟小车：
   收 ``{"target_x":x,"target_y":y}`` -> BFS 寻路 -> 回
   ``{"status":"running","path":[...]}`` 或 ``{"status":"fail","path":[]}``；
2. 联调彩排时模拟异常（不回包 / 非法 JSON / 未知状态），验证大脑端鲁棒性；
3. 单元测试里直接 import ``MockCServer`` 起一个临时端口做集成测试。

用法::

    # 正常模式，监听 9000 端口
    python tools/mock_c_server.py --port 9000

    # 模拟“障碍物堵死”
    python tools/mock_c_server.py --mode fail

    # 模拟“小车不回包”（验证超时处理）
    python tools/mock_c_server.py --mode timeout

    # 自定义障碍物（坐标为 "x,y"，分号分隔）与起点
    python tools/mock_c_server.py --obstacles "1,1;1,2;2,4" --start "0,0"

然后另一个终端运行（大脑端连过来）::

    python -m brain --transport tcp --tcp-port 9000
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Collection

# 允许直接运行本脚本（python tools/mock_c_server.py）时也能 import brain 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brain.errors import ProtocolError  # noqa: E402
from brain.grid import DEFAULT_OBSTACLES, DEFAULT_START, find_path  # noqa: E402
from brain.protocol import (  # noqa: E402
    build_fail_report,
    build_running_report,
    parse_target_message,
)

MODES = ("normal", "fail", "timeout", "bad_json", "unknown_status", "noise")


class MockCServer:
    """轻量级 TCP 服务器，行为对齐 C 端协议。

    典型用法（测试 / 脚本）::

        with MockCServer(port=0) as server:
            port = server.port
            # 用大脑端连接 127.0.0.1:port ...
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        obstacles: Collection[tuple[int, int]] | None = None,
        start: tuple[int, int] = DEFAULT_START,
        mode: str = "normal",
        delay: float = 0.0,
        verbose: bool = True,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES} 之一：{mode!r}")
        if port < 0 or port > 65535:
            raise ValueError(f"端口必须在 0~65535 之间：{port}")
        self.host = host
        self._requested_port = port
        self.obstacles: frozenset[tuple[int, int]] = (
            frozenset(obstacles) if obstacles is not None else DEFAULT_OBSTACLES
        )
        # 注意：不能叫 self.start，否则会遮蔽 start() 方法
        self.start_point = (int(start[0]), int(start[1]))
        self.mode = mode
        self.delay = max(delay, 0.0)
        self.verbose = verbose

        self._server_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._actual_port = port
        self._client_count = 0

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    @property
    def port(self) -> int:
        """实际监听端口（构造时传 0 会由系统分配，可用于测试）。"""
        return self._actual_port

    @property
    def client_count(self) -> int:
        return self._client_count

    def start(self) -> int:
        if self._thread is not None:
            return self._actual_port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self._requested_port))
        except OSError as exc:
            sock.close()
            raise OSError(f"监听 {self.host}:{self._requested_port} 失败：{exc}") from exc
        sock.listen(1)
        sock.settimeout(0.2)
        self._actual_port = sock.getsockname()[1]
        self._server_sock = sock
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._serve, name="mock-c-server", daemon=True
        )
        self._thread.start()
        return self._actual_port

    def stop(self) -> None:
        self._stop_event.set()
        sock = self._server_sock
        self._server_sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def __enter__(self) -> "MockCServer":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.stop()
        return False

    # ------------------------------------------------------------------ #
    # 内部：监听与处理
    # ------------------------------------------------------------------ #
    def _serve(self) -> None:
        sock = self._server_sock
        if sock is None:
            return
        while not self._stop_event.is_set():
            try:
                conn, addr = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._client_count += 1
            with conn:
                self._handle_client(conn, addr)

    def _handle_client(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        self._log(f"[模拟C端] 客户端已连接：{addr}")
        conn.settimeout(0.2)
        buffer = bytearray()
        try:
            while not self._stop_event.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buffer.extend(chunk)
                while b"\n" in buffer:
                    newline_index = buffer.find(b"\n")
                    raw_line = bytes(buffer[:newline_index])
                    del buffer[: newline_index + 1]
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if line:
                        self._handle_line(conn, line)
        finally:
            self._log(f"[模拟C端] 客户端已断开：{addr}")

    def _handle_line(self, conn: socket.socket, line: str) -> None:
        self._log(f"[模拟C端] 收到下发：{line}")
        if self.delay > 0:
            time.sleep(self.delay)

        if self.mode == "noise":
            # 先发一个空行（帧噪声），再正常回包：验证大脑端会跳过空行
            self._send(conn, "")
        if self.mode == "timeout":
            self._log("[模拟C端] timeout 场景：故意不回包")
            return
        if self.mode == "bad_json":
            self._send(conn, "这不是合法的 JSON 报文")
            return
        if self.mode == "unknown_status":
            self._send(conn, '{"status":"paused","path":[]}')
            return

        try:
            target = parse_target_message(line)
        except ProtocolError as exc:
            self._log(f"[模拟C端] 下发报文非法：{exc}")
            self._send(conn, build_fail_report())
            return

        path = None
        if self.mode != "fail":
            path = find_path(self.start_point, (target.x, target.y), self.obstacles)

        if path is None:
            response = build_fail_report()
        else:
            response = build_running_report(path)
        self._log(f"[模拟C端] 回包：{response}")
        self._send(conn, response)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    @staticmethod
    def _send(conn: socket.socket, text: str) -> None:
        if not text.endswith("\n"):
            text += "\n"
        try:
            conn.sendall(text.encode("utf-8"))
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_point(text: str) -> tuple[int, int]:
    parts = text.replace("，", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"坐标格式应为 'x,y'：{text!r}")
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"坐标必须是整数：{text!r}") from exc


def _parse_obstacles(text: str) -> set[tuple[int, int]]:
    obstacles: set[tuple[int, int]] = set()
    for part in text.replace("；", ";").split(";"):
        part = part.strip()
        if part:
            obstacles.add(_parse_point(part))
    return obstacles


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/mock_c_server.py",
        description="模拟 C 端（机器人躯体底层）的 TCP 服务器，用于无硬件联调彩排",
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=9000, help="监听端口（默认 9000）")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="normal",
        help="行为模式：normal/fail/timeout/bad_json/unknown_status",
    )
    parser.add_argument(
        "--obstacles",
        type=_parse_obstacles,
        default=None,
        help='障碍物坐标，如 "1,1;1,2;2,4"（默认内置一组障碍物）',
    )
    parser.add_argument(
        "--start",
        type=_parse_point,
        default=DEFAULT_START,
        help='小车起点坐标，如 "0,0"（默认左上角）',
    )
    parser.add_argument("--delay", type=float, default=0.0, help="回包前延迟（秒）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    server = MockCServer(
        host=args.host,
        port=args.port,
        obstacles=args.obstacles,
        start=args.start,
        mode=args.mode,
        delay=args.delay,
    )
    try:
        port = server.start()
    except OSError as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 2

    print("=" * 72)
    print(" 模拟 C 端 TCP 服务器已启动")
    print(f" 监听地址：{args.host}:{port}    行为模式：{args.mode}")
    print(f" 障碍物：{sorted(args.obstacles) if args.obstacles is not None else sorted(DEFAULT_OBSTACLES)}")
    print(f" 起点：{args.start}")
    print("-" * 72)
    print(" 让大脑端连过来，例如：")
    print(f"   python -m brain --transport tcp --tcp-port {port}")
    print(" 按 Ctrl+C 停止服务器。")
    print("=" * 72)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[模拟C端] 收到 Ctrl+C，正在停止 ...")
    finally:
        server.stop()
    print("[模拟C端] 已停止。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())