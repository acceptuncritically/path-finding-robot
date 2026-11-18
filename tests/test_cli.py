"""命令行入口测试：--selftest / --once / --version / 组件组装。"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from brain.cli import build_llm, main
from brain.config import AppConfig
from brain.errors import ConfigError
from brain.llm.mock import MockLLM

_MISSING_ENV_FILE = Path("这个文件不存在.env")
_MOCK_ENV = {"BRAIN_LLM_PROVIDER": "mock", "BRAIN_TRANSPORT": "mock"}


class CliTests(unittest.TestCase):
    def _run(self, argv: list[str]) -> tuple[int, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(argv)
        return code, stdout.getvalue()

    def test_selftest_passes(self) -> None:
        code, output = self._run(["--selftest"])
        self.assertEqual(code, 0)
        self.assertIn("自检结果", output)
        self.assertIn("PASS", output)

    def test_once_mock_mode_success(self) -> None:
        with mock.patch.dict(os.environ, _MOCK_ENV, clear=False):
            code, output = self._run(["--once", "去迷宫右下角"])
        self.assertEqual(code, 0)
        self.assertIn("已规划路线", output)

    def test_once_unknown_instruction_returns_nonzero(self) -> None:
        with mock.patch.dict(os.environ, _MOCK_ENV, clear=False):
            code, output = self._run(["--once", "随便走走吧"])
        self.assertEqual(code, 1)
        self.assertIn("无法确定目标", output)

    def test_version(self) -> None:
        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stdout(stdout):
            main(["--version"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("1.0.0", stdout.getvalue())

    def test_build_llm_mock(self) -> None:
        config = AppConfig.from_env(
            env={"BRAIN_LLM_PROVIDER": "mock"}, env_file=_MISSING_ENV_FILE
        )
        self.assertIsInstance(build_llm(config), MockLLM)

    def test_build_llm_openai_without_key_raises(self) -> None:
        config = AppConfig.from_env(
            env={"BRAIN_LLM_PROVIDER": "openai", "BRAIN_LLM_API_KEY": ""},
            env_file=_MISSING_ENV_FILE,
        )
        with self.assertRaises(ConfigError):
            build_llm(config)

    def test_invalid_log_file_returns_config_error_code(self) -> None:
        # 日志路径的父级是个普通文件 -> 无法创建日志，应给出友好配置错误（退出码 2）
        with tempfile.TemporaryDirectory() as tmpdir:
            blocker = Path(tmpdir) / "blocker"
            blocker.write_text("not a directory", encoding="utf-8")
            log_path = blocker / "brain.log"
            with mock.patch.dict(os.environ, _MOCK_ENV, clear=False):
                code, _ = self._run(
                    ["--once", "去右下角", "--log-file", str(log_path)]
                )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()