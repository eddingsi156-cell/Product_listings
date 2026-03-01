"""入口 — 启动 GUI 应用，桥接 asyncio 与 Qt 事件循环"""

import sys
import asyncio

try:
    import torch  # noqa: F401  — must import before PySide6 to avoid DLL conflict on Windows
except ImportError:
    pass  # torch 未安装时跳过，ML 功能在使用时再报错

from PySide6.QtWidgets import QApplication
import qasync

from .gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    # 用 qasync 将 asyncio 事件循环嵌入 Qt 事件循环
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = MainWindow()
    window.show()

    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
