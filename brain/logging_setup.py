"""日志初始化：统一格式，控制台 + 可选文件。

约定：
- 控制台默认 INFO；``-v/--verbose`` 时输出 DEBUG；
- 指定日志文件时，文件里固定记录 DEBUG（便于联调排查）；
- 所有日志都挂在 ``brain`` 这个 logger 下（各模块用 ``logging.getLogger(__name__)``）。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .errors import ConfigError

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(
    verbose: bool = False,
    log_file: str | None = None,
) -> logging.Logger:
    """初始化 ``brain`` logger；重复调用会先清理旧 handler。"""
    logger = logging.getLogger("brain")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console_handler)

    if log_file:
        path = Path(log_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(f"创建日志目录失败：{path.parent}（{exc}）") from exc
        try:
            file_handler = logging.FileHandler(path, encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"打开日志文件失败：{path}（{exc}）") from exc
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)

    return logger