"""QThread Worker 基类 — 统一取消、错误处理、生命周期管理。"""

from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class BaseWorker(QThread):
    """所有后台 Worker 的基类。

    提供:
      - ``_cancelled`` 标志位 + ``cancel()`` 方法
      - ``finished_err(str)`` 信号
      - ``run()`` 模板方法: 自动 try/except，子类只需实现 ``_run()``

    子类仍可自由声明 ``finished_ok``、``status``、``progress``
    等信号——因为各 Worker 的成功返回类型不同，不强行统一。
    """

    finished_err = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False

    def cancel(self) -> None:
        """请求取消（线程安全，仅设置标志位）。"""
        self._cancelled = True

    # ── 模板方法 ──────────────────────────────────────────────

    def run(self) -> None:
        """入口模板：调用 _run()，捕获异常发 finished_err。"""
        try:
            self._run()
        except Exception as e:
            logger.exception("Worker %s 异常", type(self).__name__)
            self.finished_err.emit(str(e))

    def _run(self) -> None:
        """子类实现此方法。成功时自行 emit finished_ok 信号。"""
        raise NotImplementedError
