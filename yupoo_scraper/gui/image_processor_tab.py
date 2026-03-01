"""图片处理标签页 — 批量 1:1 正方形补齐（原地覆盖）"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import config
from ..image_processor import (
    FolderProcessResult,
    batch_process,
)
from ..organizer import find_image_folders
from .main_image_dialog import MainImageDialog
from .base_worker import BaseWorker
from .widgets import StatusProgressBar


# ── Worker ────────────────────────────────────────────────────────

class ImageProcessWorker(BaseWorker):
    """后台批量图片处理。"""

    status = Signal(str)
    folder_started = Signal(int, int, str)     # (idx, total, folder_name)
    image_progress = Signal(int, int, str)     # (current, total, image_name)
    folder_done = Signal(int, int, object)     # (idx, total, FolderProcessResult)
    finished_ok = Signal(list)                 # list[FolderProcessResult]

    def __init__(
        self,
        folders: list[Path],
        fill_color: tuple[int, int, int],
        quality: int,
    ):
        super().__init__()
        self._folders = folders
        self._fill_color = fill_color
        self._quality = quality

    def _run(self) -> None:
        results = batch_process(
            self._folders,
            fill_color=self._fill_color,
            quality=self._quality,
            is_cancelled=lambda: self._cancelled,
            on_folder_start=lambda i, t, p: self.folder_started.emit(i, t, p.name),
            on_image_done=lambda c, t, p: self.image_progress.emit(c, t, p.name),
            on_folder_done=lambda i, t, r: self.folder_done.emit(i, t, r),
        )
        self.finished_ok.emit(results)


# ── ImageProcessorTab ─────────────────────────────────────────────

class ImageProcessorTab(QWidget):
    """图片处理标签页 — 批量 1:1 补齐（原地覆盖原图）"""

    status_message = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._download_dir: Path = config.DEFAULT_DOWNLOAD_DIR.resolve()
        self._fill_color: tuple[int, int, int] = config.SQUARE_FILL_COLOR
        self._quality: int = config.SQUARE_JPEG_QUALITY
        self._worker: ImageProcessWorker | None = None
        self._results: list[FolderProcessResult] = []
        self._build_ui()

    # ── UI 构建 ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # ── 目录选择 ──────────────────────────────────────────────
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("下载目录:"))
        self._dir_label = QLabel(str(self._download_dir))
        self._dir_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        btn_browse = QPushButton("浏览...")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._on_browse)
        dir_layout.addWidget(self._dir_label, 1)
        dir_layout.addWidget(btn_browse)
        main_layout.addLayout(dir_layout)

        # ── 参数栏 ──────────────────────────────────────────────
        param_layout = QHBoxLayout()

        # 填充颜色
        param_layout.addWidget(QLabel("填充颜色:"))
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(28, 28)
        self._update_color_btn()
        self._color_btn.clicked.connect(self._on_pick_color)
        param_layout.addWidget(self._color_btn)

        param_layout.addSpacing(16)

        # JPEG 质量
        param_layout.addWidget(QLabel("JPEG 质量:"))
        self._quality_slider = QSlider(Qt.Orientation.Horizontal)
        self._quality_slider.setMinimum(50)
        self._quality_slider.setMaximum(100)
        self._quality_slider.setValue(self._quality)
        self._quality_slider.setTickInterval(5)
        self._quality_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._quality_slider.setFixedWidth(150)
        self._quality_slider.valueChanged.connect(self._on_quality_changed)
        param_layout.addWidget(self._quality_slider)

        self._quality_label = QLabel(str(self._quality))
        self._quality_label.setFixedWidth(30)
        param_layout.addWidget(self._quality_label)

        param_layout.addStretch()
        main_layout.addLayout(param_layout)

        # ── 控制栏 ──────────────────────────────────────────────
        ctrl_layout = QHBoxLayout()

        self._btn_process = QPushButton("开始处理")
        self._btn_process.clicked.connect(self._on_process)
        ctrl_layout.addWidget(self._btn_process)

        self._btn_stop = QPushButton("停止")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        ctrl_layout.addWidget(self._btn_stop)

        ctrl_layout.addStretch()

        self._summary_label = QLabel("")
        ctrl_layout.addWidget(self._summary_label)

        main_layout.addLayout(ctrl_layout)

        # ── 进度条 ──────────────────────────────────────────────
        self._progress = StatusProgressBar()
        main_layout.addWidget(self._progress)

        # ── 结果表格 ────────────────────────────────────────────
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["状态", "文件夹名称", "图片数", "已补齐/已跳过/失败"]
        )
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        self._table.doubleClicked.connect(self._on_row_double_clicked)
        main_layout.addWidget(self._table, 1)

    # ── 目录浏览 ──────────────────────────────────────────────────

    @Slot()
    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择下载目录", str(self._download_dir),
        )
        if path:
            self._download_dir = Path(path)
            self._dir_label.setText(str(self._download_dir))

    # ── 参数控制 ──────────────────────────────────────────────────

    def _update_color_btn(self) -> None:
        r, g, b = self._fill_color
        self._color_btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #888;"
        )

    @Slot()
    def _on_pick_color(self) -> None:
        r, g, b = self._fill_color
        color = QColorDialog.getColor(QColor(r, g, b), self, "选择填充颜色")
        if color.isValid():
            self._fill_color = (color.red(), color.green(), color.blue())
            self._update_color_btn()

    @Slot(int)
    def _on_quality_changed(self, value: int) -> None:
        self._quality = value
        self._quality_label.setText(str(value))

    # ── 处理控制 ──────────────────────────────────────────────────

    @Slot()
    def _on_process(self) -> None:
        if not self._download_dir.exists():
            self._progress.set_status("目录不存在")
            self.status_message.emit("目录不存在")
            return

        folders = self._find_image_folders(self._download_dir)
        if not folders:
            self._progress.set_status("未找到含图片的子文件夹")
            self.status_message.emit("未找到含图片的文件夹")
            return

        # 重置状态
        self._results.clear()
        self._table.setRowCount(0)
        self._summary_label.setText("")

        self._btn_process.setEnabled(False)
        self._btn_stop.setEnabled(True)

        self._worker = ImageProcessWorker(
            folders,
            fill_color=self._fill_color,
            quality=self._quality,
        )
        self._worker.status.connect(self._on_worker_status)
        self._worker.folder_started.connect(self._on_folder_started)
        self._worker.image_progress.connect(self._on_image_progress)
        self._worker.folder_done.connect(self._on_folder_done)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.finished_err.connect(self._on_error)
        self._worker.start()

    @Slot()
    def _on_stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            self._btn_stop.setEnabled(False)
            self._progress.set_status("正在停止...")

    # ── Worker 信号处理 ───────────────────────────────────────────

    @Slot(str)
    def _on_worker_status(self, msg: str) -> None:
        self._progress.set_status(msg)

    @Slot(int, int, str)
    def _on_folder_started(self, idx: int, total: int, name: str) -> None:
        self._progress.set_status(f"正在处理: {name}")
        self._progress.set_progress(idx, total)

    @Slot(int, int, str)
    def _on_image_progress(self, current: int, total: int, name: str) -> None:
        self._progress.set_stats(f"图片 {current}/{total}")

    @Slot(int, int, object)
    def _on_folder_done(self, idx: int, total: int, result: FolderProcessResult) -> None:
        self._results.append(result)
        self._add_table_row(result)
        self._progress.set_progress(idx + 1, total)
        self._update_summary()

    @Slot(list)
    def _on_finished(self, results: list) -> None:
        self._btn_process.setEnabled(True)
        self._btn_stop.setEnabled(False)
        n = len(self._results)
        total_proc = sum(r.processed for r in self._results)
        total_skip = sum(r.skipped for r in self._results)
        total_fail = sum(r.failed for r in self._results)
        msg = f"处理完成: {n} 个文件夹, {total_proc} 张补齐, {total_skip} 张跳过"
        if total_fail:
            msg += f", {total_fail} 张失败"
        self._progress.set_status(msg)
        self._progress.set_stats("")
        self._progress.set_progress(n, n)
        self.status_message.emit(msg)
        self._worker = None

    @Slot(str)
    def _on_error(self, error: str) -> None:
        self._btn_process.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.set_status(f"处理出错: {error}")
        self.status_message.emit(f"处理出错: {error}")
        self._worker = None

    # ── 表格操作 ──────────────────────────────────────────────────

    def _add_table_row(self, result: FolderProcessResult) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        # 状态列
        status_item = QTableWidgetItem()
        status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if result.errors:
            status_item.setText("!")
            status_item.setToolTip("\n".join(result.errors))
            status_item.setForeground(Qt.GlobalColor.darkYellow)
        elif result.total == 0:
            status_item.setText("—")
            status_item.setToolTip("无图片")
        else:
            status_item.setText("OK")
            status_item.setToolTip("处理成功")
            status_item.setForeground(Qt.GlobalColor.darkGreen)
        self._table.setItem(row, 0, status_item)

        # 文件夹名称
        name_item = QTableWidgetItem(result.folder.name)
        self._table.setItem(row, 1, name_item)

        # 图片数
        count_item = QTableWidgetItem(str(result.total))
        count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, 2, count_item)

        # 已补齐/已跳过/失败
        detail = f"{result.processed}/{result.skipped}/{result.failed}"
        detail_item = QTableWidgetItem(detail)
        detail_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if result.failed > 0:
            detail_item.setForeground(Qt.GlobalColor.red)
        self._table.setItem(row, 3, detail_item)

    def _update_summary(self) -> None:
        n = len(self._results)
        total_imgs = sum(r.total for r in self._results)
        total_proc = sum(r.processed for r in self._results)
        self._summary_label.setText(
            f"共 {n} 个文件夹, {total_imgs} 张图片, {total_proc} 张已补齐"
        )

    # ── 主图选择 ──────────────────────────────────────────────────

    @Slot()
    def _on_row_double_clicked(self, index) -> None:
        row = index.row()
        if row < 0 or row >= len(self._results):
            return
        folder = self._results[row].folder
        if not folder.exists():
            return
        dlg = MainImageDialog(folder, parent=self)
        dlg.exec()

    # ── 辅助方法 ──────────────────────────────────────────────────

    @staticmethod
    def _find_image_folders(base_dir: Path) -> list[Path]:
        return find_image_folders(base_dir)

    def cleanup(self) -> None:
        """安全停止后台线程。"""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)

    def set_download_dir(self, path: Path) -> None:
        """外部设置下载目录（如同步采集页路径）。"""
        self._download_dir = path
        self._dir_label.setText(str(path))
