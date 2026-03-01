"""自定义 GUI 组件"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QWidget,
)


class StatusProgressBar(QWidget):
    """带文字说明的进度条组件。

    布局: [状态文字] [████████░░░ 60%] [统计信息]
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._status_label = QLabel("就绪")
        self._status_label.setMinimumWidth(200)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setMinimumWidth(200)

        self._stats_label = QLabel("")
        self._stats_label.setFixedWidth(200)
        self._stats_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        layout.addWidget(self._status_label)
        layout.addWidget(self._progress_bar, 1)
        layout.addWidget(self._stats_label)

    def set_status(self, text: str):
        self._status_label.setText(text)

    def set_progress(self, current: int, total: int):
        if total > 0:
            pct = int(current / total * 100)
            self._progress_bar.setValue(pct)
            self._stats_label.setText(f"{current}/{total}")
        else:
            self._progress_bar.setValue(0)
            self._stats_label.setText("")

    def set_stats(self, text: str):
        self._stats_label.setText(text)

    def reset(self):
        self._progress_bar.setValue(0)
        self._status_label.setText("就绪")
        self._stats_label.setText("")
