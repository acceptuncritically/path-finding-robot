"""TCP 传输层：一行一条 JSON（``\\n`` 分帧），带粘包/半包处理。

适用场景：前期在电脑上做仿真联调（C 端程序或 ``tools/mock_c_server.py``
监听 TCP 端口，Python 大脑端作为客户端连接）。
与串口版本共用同一套 ``send_line`` / ``recv_line`` 接口，业务层不感知差异。
"""

from __future__ import annotations

import logging
import select
import socket
import time

from ..errors import TransportClosedError, TransportError
from .base import Transport

logger = logging.getLogger(__name__)

_MAX_LINE_BYTES = 1 << 20  # 单条报文上限 1 MiB，防止异常数据打爆内存


class TcpTransport(Transport):
    """TCP 客户端传输层。"""

    name = "tcp"

    def __init__(
        self,
        host: str,
        port: int,
        connect_timeout: float = 3.0,
        max_line_bytes: int = _MAX_LINE_BYTES,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.connect_timeout = connect_timeout
        self.max_line_bytes = max_line_bytes
        self._sock: socket.socket | None = None
        self._buffer = bytearray()

    def describe(self) -> str:
        return f"TCP 客户端 -> {self.host}:{self.port}"

    @property
    def is_open(self) -> bool:
        return self._sock is not None

    # ------------------------------------------------------------------ #
    # 连接管理
    # ------------------------------------------------------------------ #
    def open(self) -> None:
        if self._sock is not None:
            return
        try:
            sock = socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout
            )
        except OSError as exc:
            raise TransportError(
                f"连接 TCP 服务器 {self.host}:{self.port} 失败：{exc}"
            ) from exc
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # 接收时用 select 控制超时，socket 本身设为阻塞即可
        sock.setblocking(True)
        self._sock = sock
        self._buffer.clear()
        logger.info("TCP 已连接：%s:%s", self.host, self.port)

    def close(self) -> None:
        sock = self._sock
        self._sock = None
        self._buffer.clear()
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
        logger.info("TCP 已断开：%s:%s", self.host, self.port)

    # ------------------------------------------------------------------ #
    # 收发
    # ------------------------------------------------------------------ #
    def send_line(self, text: str) -> None:
        sock = self._require_socket()
        if not text.endswith("\n"):
            text += "\n"
        try:
            sock.sendall(text.encode("utf-8"))
        except OSError as exc:
            self._mark_broken()
            raise TransportError(f"TCP 发送失败：{exc}") from exc
        logger.debug("TCP 已发送：%r", text)

    def recv_line(self, timeout: float) -> str | None:
        sock = self._require_socket()
        deadline = time.monotonic() + max(timeout, 0.0)

        while True:
            line = self._pop_buffered_line()
            if line is not None:
                if not line.strip():
                    logger.debug("TCP 跳过空行（帧噪声）")
                    continue
                logger.debug("TCP 收到：%r", line)
                return line

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None

            try:
                readable, _, _ = select.select([sock], [], [], remaining)
            except OSError as exc:
                raise TransportError(f"TCP 等待数据失败：{exc}") from exc
            if not readable:
                return None

            try:
                chunk = sock.recv(4096)
            except OSError as exc:
                raise TransportError(f"TCP 接收失败：{exc}") from exc
            if not chunk:
                self._mark_broken()
                raise TransportClosedError("TCP 连接已被对端关闭")
            self._buffer.extend(chunk)
            if len(self._buffer) > self.max_line_bytes:
                raise TransportError(
                    f"单条报文超过 {self.max_line_bytes} 字节上限，疑似协议错乱"
                )

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _require_socket(self) -> socket.socket:
        if self._sock is None:
            raise TransportError("TCP 连接尚未建立，请先调用 open()")
        return self._sock

    def _pop_buffered_line(self) -> str | None:
        newline_index = self._buffer.find(b"\n")
        if newline_index < 0:
            return None
        raw_line = bytes(self._buffer[:newline_index])
        del self._buffer[: newline_index + 1]
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TransportError(f"TCP 收到非 UTF-8 报文：{exc}") from exc
        return line.rstrip("\r")

    def _mark_broken(self) -> None:
        """标记连接已失效并及时释放 socket（避免误以为还能继续收发）。"""
        sock = self._sock
        self._sock = None
        self._buffer.clear()
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass