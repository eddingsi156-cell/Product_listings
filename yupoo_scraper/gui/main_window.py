"""主窗口 — 菜单栏、状态栏、标签页布局"""

from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from .dedup_tab import DedupTab
from .image_processor_tab import ImageProcessorTab
from .processor_tab import ProcessorTab
from .scraper_tab import ScraperTab
from .uploader_tab import UploaderTab


class MainWindow(QMainWindow):
    """应用主窗口"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("产品上架工具")
        self.setMinimumSize(900, 650)
        self.resize(1000, 700)

        # 标签页
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        # 采集页
        self._scraper_tab = ScraperTab()
        self._tabs.addTab(self._scraper_tab, "采集")

        # 拆分页
        self._processor_tab = ProcessorTab()
        self._tabs.addTab(self._processor_tab, "拆分")

        # 查重页
        self._dedup_tab = DedupTab()
        self._tabs.addTab(self._dedup_tab, "查重")

        # 图片处理页
        self._image_proc_tab = ImageProcessorTab()
        self._tabs.addTab(self._image_proc_tab, "图片处理")

        # 上架页
        self._uploader_tab = UploaderTab()
        self._tabs.addTab(self._uploader_tab, "上架")

        # 状态栏
        status_bar = QStatusBar()
        self._status_label = QLabel("就绪")
        status_bar.addWidget(self._status_label, 1)
        self._version_label = QLabel("v0.5.0")
        status_bar.addPermanentWidget(self._version_label)
        self.setStatusBar(status_bar)

        # 连接信号
        self._scraper_tab.status_message.connect(self._on_status_message)
        self._processor_tab.status_message.connect(self._on_status_message)
        self._dedup_tab.status_message.connect(self._on_status_message)
        self._image_proc_tab.status_message.connect(self._on_status_message)
        self._uploader_tab.status_message.connect(self._on_status_message)

    @Slot(str)
    def _on_status_message(self, msg: str):
        self._status_label.setText(msg)

    def closeEvent(self, event):
        self._scraper_tab.cleanup()
        self._processor_tab.cleanup()
        self._dedup_tab.cleanup()
        self._image_proc_tab.cleanup()
        self._uploader_tab.cleanup()
        super().closeEvent(event)
