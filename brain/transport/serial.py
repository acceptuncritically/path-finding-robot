"""串口传输层：一行一条 JSON（``\\n`` 分帧）。

为什么串口优先：最终要烧录到小车硬件上，电脑与小车之间最直接的连接就是串口。
本模块只在 ``open()`` 时延迟导入 ``pyserial``——没装依赖时核心功能完全不受影响，
并给出明确的安装提示。

约定（联调时与 C 端确认）：
- 波特率默认 115200、8N1；
- 每条报文 UTF-8 编码、以 ``\\n`` 结尾；
- 接收按行缓冲，收到完整一行才返回。
"""

from __future__ import annotations

import logging
import time

from ..errors import TransportClosedError, TransportError
from .base import Transport

logger = logging.getLogger(__name__)

_MAX_LINE_BYTES = 1 << 20


class SerialTransport(Transport):
    """串口传输层（依赖可选的 ``pyserial``）。"""

    name = "serial"

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        read_timeout: float = 0.05,
        max_line_bytes: int = _MAX_LINE_BYTES,
    ) -> None:
        self.port = port
        self.baudrate = int(baudrate)
        self.read_timeout = read_timeout
        self.max_line_bytes = max_line_bytes
        self._serial = None
        self._buffer = bytearray()

    def describe(self) -> str:
        return f"串口 {self.port}（{self.baudrate} bps, 8N1）"

    @property
    def is_open(self) -> bool:
        return self._serial is not None

    # ------------------------------------------------------------------ #
    # 连接管理
    # ------------------------------------------------------------------ #
    def open(self) -> None:
        if self._serial is not None:
            return
        try:
            import serial  # 延迟导入：核心功能不依赖 pyserial
        except ImportError as exc:
            raise TransportError(
                "串口功能需要 pyserial，请先执行：pip install pyserial"
            ) from exc
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.read_timeout,
                write_timeout=1.0,
            )
        except (serial.SerialException, OSError) as exc:
            raise TransportError(f"打开串口 {self.port} 失败：{exc}") from exc
        self._buffer.clear()
        logger.info("串口已打开：%s（%d bps）", self.port, self.baudrate)

    def close(self) -> None:
        serial_obj = self._serial
        self._serial = None
        self._buffer.clear()
        if serial_obj is None:
            return
        try:
            serial_obj.close()
        except OSError:
            pass
        logger.info("串口已关闭：%s", self.port)

    # ------------------------------------------------------------------ #
    # 收发
    # ------------------------------------------------------------------ #
    def send_line(self, text: str) -> None:
        serial_obj = self._require_serial()
        if not text.endswith("\n"):
            text += "\n"
        try:
            serial_obj.write(text.encode("utf-8"))
            serial_obj.flush()
        except Exception as exc:  # pyserial 的 SerialException / 底层 OSError
            self._mark_broken()
            raise TransportError(f"串口发送失败：{exc}") from exc
        logger.debug("串口已发送：%r", text)

    def recv_line(self, timeout: float) -> str | None:
        serial_obj = self._require_serial()
        deadline = time.monotonic() + max(timeout, 0.0)

        while True:
            line = self._pop_buffered_line()
            if line is not None:
                if not line.strip():
                    logger.debug("串口跳过空行（帧噪声）")
                    continue
                logger.debug("串口收到：%r", line)
                return line

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None

            try:
                chunk = serial_obj.read(1)
            except Exception as exc:  # pyserial 的 SerialException
                self._mark_broken()
                raise TransportError(f"串口接收失败：{exc}") from exc
            if chunk:
                self._buffer.extend(chunk)
                if len(self._buffer) > self.max_line_bytes:
                    raise TransportError(
                        f"单条报文超过 {self.max_line_bytes} 字节上限，疑似协议错乱"
                    )
                continue
            # read 超时返回 b""：循环继续，直到总 deadline

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _require_serial(self):
        if self._serial is None:
            raise TransportError("串口尚未打开，请先调用 open()")
        return self._serial

    def _pop_buffered_line(self) -> str | None:
        newline_index = self._buffer.find(b"\n")
        if newline_index < 0:
            return None
        raw_line = bytes(self._buffer[:newline_index])
        del self._buffer[: newline_index + 1]
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TransportError(f"串口收到非 UTF-8 报文：{exc}") from exc
        return line.rstrip("\r")

    def _mark_broken(self) -> None:
        """标记串口已失效并及时关闭句柄。"""
        serial_obj = self._serial
        self._serial = None
        self._buffer.clear()
        if serial_obj is not None:
            try:
                serial_obj.close()
            except Exception:  # noqa: BLE001 - 关闭失败不影响主流程
                pass