"""入口 — 启动 GUI 应用，桥接 asyncio 与 Qt 事件循环"""

import logging
import sys
import asyncio

try:
    import torch  # noqa: F401  — must import before PySide6 to avoid DLL conflict on Windows
except ImportError:
    pass  # torch 未安装时跳过，ML 功能在使用时再报错

from PySide6.QtWidgets import QApplication, QMessageBox
import qasync

from .gui.main_window import MainWindow

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """配置日志：控制台输出 + 统一格式。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # 降低第三方库的日志级别
    for noisy in ("PIL", "urllib3", "asyncio", "aiohttp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _install_excepthook() -> None:
    """安装全局未捕获异常处理器，记录日志并弹窗提示。"""
    _original_hook = sys.excepthook

    def _handler(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            _original_hook(exc_type, exc_value, exc_tb)
            return
        logger.critical("未捕获异常", exc_info=(exc_type, exc_value, exc_tb))
        # 尝试弹窗（可能在非 GUI 线程中，忽略错误）
        try:
            QMessageBox.critical(
                None, "程序错误",
                f"发生未捕获的异常：\n{exc_type.__name__}: {exc_value}\n\n"
                "详细信息已记录到日志。",
            )
        except Exception:
            pass

    sys.excepthook = _handler


def main() -> None:
    _setup_logging()
    _install_excepthook()

    app = QApplication(sys.argv)

    # 用 qasync 将 asyncio 事件循环嵌入 Qt 事件循环
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = MainWindow()
    window.show()

    logger.info("应用启动")
    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
