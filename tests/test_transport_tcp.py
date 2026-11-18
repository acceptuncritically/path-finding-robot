"""TCP 传输层集成测试：真实 socket + 模拟 C 端服务器（不需要硬件）。

这些测试验证“第 3 步 联调”的核心链路：TCP 收发、按行分帧、
超时、坏报文、对端断开，全部在本地回环完成。
"""

from __future__ import annotations

import unittest

from brain.errors import MalformedReportError, TransportClosedError, TransportError
from brain.protocol import Target, build_target_message, parse_robot_report
from brain.transport.tcp import TcpTransport
from tools.mock_c_server import MockCServer


class TcpTransportIntegrationTests(unittest.TestCase):
    def _start_server(self, **kwargs: object) -> tuple[MockCServer, int]:
        server = MockCServer(
            host="127.0.0.1", port=0, verbose=False, **kwargs  # type: ignore[arg-type]
        )
        port = server.start()
        self.addCleanup(server.stop)
        return server, port

    def _connect(self, port: int) -> TcpTransport:
        transport = TcpTransport("127.0.0.1", port, connect_timeout=2.0)
        transport.open()
        self.addCleanup(transport.close)
        return transport

    def test_round_trip_running(self) -> None:
        _, port = self._start_server()
        transport = self._connect(port)

        transport.send_line(build_target_message(Target(5, 5)))
        raw = transport.recv_line(5.0)

        self.assertIsNotNone(raw)
        report = parse_robot_report(raw)  # type: ignore[arg-type]
        self.assertEqual(report.status, "running")
        self.assertEqual((report.path[-1].x, report.path[-1].y), (5, 5))

    def test_round_trip_fail(self) -> None:
        _, port = self._start_server(mode="fail")
        transport = self._connect(port)

        transport.send_line(build_target_message(Target(5, 5)))
        raw = transport.recv_line(5.0)

        report = parse_robot_report(raw)  # type: ignore[arg-type]
        self.assertEqual(report.status, "fail")
        self.assertEqual(report.path, ())

    def test_timeout_returns_none(self) -> None:
        _, port = self._start_server(mode="timeout")
        transport = self._connect(port)

        transport.send_line(build_target_message(Target(3, 2)))
        self.assertIsNone(transport.recv_line(0.3))

    def test_bad_json_from_c_side(self) -> None:
        _, port = self._start_server(mode="bad_json")
        transport = self._connect(port)

        transport.send_line(build_target_message(Target(3, 2)))
        raw = transport.recv_line(5.0)
        with self.assertRaises(MalformedReportError):
            parse_robot_report(raw)  # type: ignore[arg-type]

    def test_skips_empty_line_noise(self) -> None:
        # 模拟 C 端在真正回包前多发一个空行（帧噪声）：应跳过并拿到真正回包
        _, port = self._start_server(mode="noise")
        transport = self._connect(port)

        transport.send_line(build_target_message(Target(5, 5)))
        raw = transport.recv_line(5.0)

        self.assertIsNotNone(raw)
        report = parse_robot_report(raw)  # type: ignore[arg-type]
        self.assertEqual(report.status, "running")

    def test_connect_refused_raises_transport_error(self) -> None:
        # 起一个服务器拿到端口后立刻停掉，确保这个端口没有监听
        server = MockCServer(host="127.0.0.1", port=0, verbose=False)
        port = server.start()
        server.stop()

        transport = TcpTransport("127.0.0.1", port, connect_timeout=0.5)
        with self.assertRaises(TransportError):
            transport.open()

    def test_detects_peer_closed(self) -> None:
        server, port = self._start_server()
        transport = self._connect(port)

        transport.send_line(build_target_message(Target(0, 0)))
        self.assertIsNotNone(transport.recv_line(5.0))

        server.stop()
        with self.assertRaises(TransportClosedError):
            transport.recv_line(2.0)

        # 断开后应如实标记连接已失效，后续发送直接报错而不是静默失败
        self.assertFalse(transport.is_open)
        with self.assertRaises(TransportError):
            transport.send_line(build_target_message(Target(0, 0)))

    def test_send_before_open_raises(self) -> None:
        transport = TcpTransport("127.0.0.1", 1)
        with self.assertRaises(TransportError):
            transport.send_line('{"target_x":0,"target_y":0}')


if __name__ == "__main__":
    unittest.main()