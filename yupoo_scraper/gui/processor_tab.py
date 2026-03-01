"""处理标签页 — 批量扫描 + 结果表格 + 审核联动"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFileDialog,
    QGroupBox,
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
from ..config import (
    CLUSTER_THRESHOLD_DEFAULT,
    CLUSTER_THRESHOLD_MAX,
    CLUSTER_THRESHOLD_MIN,
)
from ..ml.splitter import (
    BatchScanItem,
    SplitResult,
    batch_extract_and_split,
    recluster,
)
from ..organizer import find_image_folders
from .split_dialog import SplitDialog
from .widgets import StatusProgressBar


# ── BatchScanWorker ────────────────────────────────────────────

class BatchScanWorker(QThread):
    """后台批量扫描：逐文件夹提取特征+聚类。"""

    status = Signal(str)
    folder_started = Signal(int, int, str)   # (idx, total, folder_name)
    image_progress = Signal(int, int)        # (current, total)
    folder_done = Signal(int, int, object)   # (idx, total, BatchScanItem)
    finished_ok = Signal(list)               # list[BatchScanItem]
    finished_err = Signal(str)

    def __init__(self, folders: list[Path], threshold: float):
        super().__init__()
        self._folders = folders
        self._threshold = threshold
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            results = batch_extract_and_split(
                self._folders,
                self._threshold,
                is_cancelled=lambda: self._cancelled,
                on_folder_start=lambda i, t, p: self.folder_started.emit(i, t, p.name),
                on_status=lambda msg: self.status.emit(msg),
                on_image_progress=lambda c, t: self.image_progress.emit(c, t),
                on_folder_done=lambda i, t, item: self.folder_done.emit(i, t, item),
            )
            self.finished_ok.emit(results)
        except Exception as e:
            self.finished_err.emit(str(e))


# ── 筛选枚举 ───────────────────────────────────────────────────

_FILTER_ALL = 0
_FILTER_NEED_SPLIT = 1
_FILTER_NO_SPLIT = 2


# ── ProcessorTab ───────────────────────────────────────────────

class ProcessorTab(QWidget):
    """处理标签页 — 批量扫描所有相册 → 结果表格 → 审核拆分"""

    status_message = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._download_dir: Path = config.DEFAULT_DOWNLOAD_DIR.resolve()
        self._worker: BatchScanWorker | None = None
        self._scan_results: list[BatchScanItem] = []
        self._current_filter = _FILTER_ALL
        self._recluster_timer = QTimer(self)
        self._recluster_timer.setSingleShot(True)
        self._recluster_timer.setInterval(300)
        self._recluster_timer.timeout.connect(self._do_recluster)
        self._build_ui()

    # ── UI 构建 ─────────────────────────────────────────────────

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # ── 目录选择 ──────────────────────────────────────────
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

        # ── 扫描控制栏 ────────────────────────────────────────
        ctrl_layout = QHBoxLayout()

        self._btn_scan = QPushButton("扫描全部")
        self._btn_scan.clicked.connect(self._on_scan)
        ctrl_layout.addWidget(self._btn_scan)

        self._btn_stop = QPushButton("停止")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        ctrl_layout.addWidget(self._btn_stop)

        ctrl_layout.addSpacing(16)
        ctrl_layout.addWidget(QLabel("灵敏度:"))

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(int(CLUSTER_THRESHOLD_MIN * 100))
        self._slider.setMaximum(int(CLUSTER_THRESHOLD_MAX * 100))
        self._slider.setValue(int(CLUSTER_THRESHOLD_DEFAULT * 100))
        self._slider.setTickInterval(5)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setFixedWidth(180)
        self._slider.valueChanged.connect(self._on_slider_changed)
        ctrl_layout.addWidget(self._slider)

        self._threshold_label = QLabel(f"{CLUSTER_THRESHOLD_DEFAULT:.2f}")
        self._threshold_label.setFixedWidth(36)
        ctrl_layout.addWidget(self._threshold_label)
        ctrl_layout.addStretch()

        main_layout.addLayout(ctrl_layout)

        # ── 进度条 ────────────────────────────────────────────
        self._progress = StatusProgressBar()
        main_layout.addWidget(self._progress)

        # ── 筛选栏 ────────────────────────────────────────────
        filter_layout = QHBoxLayout()

        self._btn_filter_all = QPushButton("全部")
        self._btn_filter_need = QPushButton("需拆分")
        self._btn_filter_ok = QPushButton("无需拆分")

        for btn in (self._btn_filter_all, self._btn_filter_need, self._btn_filter_ok):
            btn.setCheckable(True)
            btn.setFixedWidth(80)
        self._btn_filter_all.setChecked(True)

        self._filter_group = QButtonGroup(self)
        self._filter_group.setExclusive(True)
        self._filter_group.addButton(self._btn_filter_all, _FILTER_ALL)
        self._filter_group.addButton(self._btn_filter_need, _FILTER_NEED_SPLIT)
        self._filter_group.addButton(self._btn_filter_ok, _FILTER_NO_SPLIT)
        self._filter_group.idClicked.connect(self._on_filter_changed)

        filter_layout.addWidget(self._btn_filter_all)
        filter_layout.addWidget(self._btn_filter_need)
        filter_layout.addWidget(self._btn_filter_ok)
        filter_layout.addStretch()

        self._summary_label = QLabel("")
        filter_layout.addWidget(self._summary_label)

        main_layout.addLayout(filter_layout)

        # ── 结果表格 ──────────────────────────────────────────
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["状态", "文件夹名称", "图片数", "分组数", "操作"]
        )
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        main_layout.addWidget(self._table, 1)

    # ── 目录浏览 ───────────────────────────────────────────────

    @Slot()
    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择下载目录", str(self._download_dir),
        )
        if path:
            self._download_dir = Path(path)
            self._dir_label.setText(str(self._download_dir))

    # ── 扫描控制 ───────────────────────────────────────────────

    @Slot()
    def _on_scan(self) -> None:
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
        self._scan_results.clear()
        self._table.setRowCount(0)
        self._update_summary()

        threshold = self._slider.value() / 100.0

        self._btn_scan.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._slider.setEnabled(False)

        self._worker = BatchScanWorker(folders, threshold)
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

    # ── Worker 信号处理 ─────────────────────────────────────────

    @Slot(str)
    def _on_worker_status(self, msg: str) -> None:
        self._progress.set_status(msg)

    @Slot(int, int, str)
    def _on_folder_started(self, idx: int, total: int, name: str) -> None:
        self._progress.set_status(f"正在处理: {name}")
        self._progress.set_progress(idx, total)

    @Slot(int, int)
    def _on_image_progress(self, current: int, total: int) -> None:
        # 图片级进度显示在 stats 里
        self._progress.set_stats(f"图片 {current}/{total}")

    @Slot(int, int, object)
    def _on_folder_done(self, idx: int, total: int, item: BatchScanItem) -> None:
        self._scan_results.append(item)
        self._add_table_row(item)
        self._progress.set_progress(idx + 1, total)
        self._update_summary()

    @Slot(list)
    def _on_finished(self, results: list) -> None:
        self._btn_scan.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._slider.setEnabled(True)
        n = len(self._scan_results)
        need = sum(1 for r in self._scan_results if r.group_count > 1)
        self._progress.set_status(f"扫描完成: {n} 个文件夹, {need} 个需拆分")
        self._progress.set_stats("")
        self._progress.set_progress(n, n)
        self.status_message.emit(f"扫描完成: {n} 个文件夹, {need} 个需拆分")
        self._worker = None

    @Slot(str)
    def _on_error(self, error: str) -> None:
        self._btn_scan.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._slider.setEnabled(True)
        self._progress.set_status(f"扫描出错: {error}")
        self.status_message.emit(f"扫描出错: {error}")
        self._worker = None

    # ── 表格操作 ───────────────────────────────────────────────

    def _add_table_row(self, item: BatchScanItem) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        # 状态列
        status_item = QTableWidgetItem()
        status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_status_cell(status_item, item)
        self._table.setItem(row, 0, status_item)

        # 文件夹名称
        name_item = QTableWidgetItem(item.folder.name)
        self._table.setItem(row, 1, name_item)

        # 图片数
        count_item = QTableWidgetItem(str(item.image_count))
        count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, 2, count_item)

        # 分组数
        group_item = QTableWidgetItem(str(item.group_count))
        group_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, 3, group_item)

        # 操作列 — 审核按钮或占位
        if item.error:
            btn = QLabel("出错")
            btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
            btn.setStyleSheet("color: red;")
            btn.setToolTip(item.error)
        elif item.group_count > 1:
            btn = QPushButton("审核")
            btn.clicked.connect(lambda _=None, r=row: self._on_review(r))
        else:
            btn = QLabel("—")
            btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setCellWidget(row, 4, btn)

    def _update_status_cell(self, cell: QTableWidgetItem, item: BatchScanItem) -> None:
        if item.error:
            cell.setText("X")
            cell.setToolTip(item.error)
            cell.setForeground(Qt.GlobalColor.red)
        elif item.group_count > 1:
            cell.setText("!")
            cell.setToolTip("需要拆分")
            cell.setForeground(Qt.GlobalColor.darkYellow)
        else:
            cell.setText("OK")
            cell.setToolTip("无需拆分")
            cell.setForeground(Qt.GlobalColor.darkGreen)

    def _refresh_table_row(self, row: int) -> None:
        """刷新指定行的数据（灵敏度变化或拆分完成后）。"""
        item = self._scan_results[row]

        # 状态
        status_cell = self._table.item(row, 0)
        self._update_status_cell(status_cell, item)

        # 分组数
        self._table.item(row, 3).setText(str(item.group_count))

        # 操作列
        if item.error:
            widget = QLabel("出错")
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            widget.setStyleSheet("color: red;")
            widget.setToolTip(item.error)
        elif item.group_count > 1:
            btn = QPushButton("审核")
            btn.clicked.connect(lambda _=None, r=row: self._on_review(r))
            widget = btn
        else:
            widget = QLabel("—")
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setCellWidget(row, 4, widget)

        self._apply_row_filter(row)

    # ── 灵敏度滑块 ─────────────────────────────────────────────

    @Slot(int)
    def _on_slider_changed(self, value: int) -> None:
        threshold = value / 100.0
        self._threshold_label.setText(f"{threshold:.2f}")
        if self._scan_results:
            self._recluster_timer.start()  # 防抖：300ms 内无新变化才执行

    def _do_recluster(self) -> None:
        """防抖后实际执行重聚类。"""
        from PySide6.QtWidgets import QApplication

        # 禁用滑块防止重入（processEvents 期间用户可能继续拖动）
        self._slider.setEnabled(False)
        try:
            threshold = self._slider.value() / 100.0
            for i, item in enumerate(self._scan_results):
                if item.error or item.image_count == 0:
                    continue
                new_result = recluster(item.result, threshold)
                self._scan_results[i] = BatchScanItem(
                    folder=item.folder,
                    image_count=item.image_count,
                    group_count=len(new_result.groups),
                    result=new_result,
                    error=None,
                )
                self._refresh_table_row(i)
                QApplication.processEvents()  # 避免大量结果时 UI 冻结
            self._update_summary()
        finally:
            self._slider.setEnabled(True)

    # ── 筛选 ───────────────────────────────────────────────────

    @Slot(int)
    def _on_filter_changed(self, filter_id: int) -> None:
        self._current_filter = filter_id
        for row in range(self._table.rowCount()):
            self._apply_row_filter(row)

    def _apply_row_filter(self, row: int) -> None:
        if row >= len(self._scan_results):
            return
        item = self._scan_results[row]
        needs_split = item.group_count > 1

        if self._current_filter == _FILTER_ALL:
            self._table.setRowHidden(row, False)
        elif self._current_filter == _FILTER_NEED_SPLIT:
            self._table.setRowHidden(row, not needs_split)
        else:  # _FILTER_NO_SPLIT
            self._table.setRowHidden(row, needs_split)

    def _update_summary(self) -> None:
        total = len(self._scan_results)
        need = sum(1 for r in self._scan_results if r.group_count > 1)
        self._summary_label.setText(f"共 {total} 个, {need} 个需拆分")

    # ── 审核 ───────────────────────────────────────────────────

    def _on_review(self, row: int) -> None:
        if row >= len(self._scan_results):
            return
        item = self._scan_results[row]
        threshold = self._slider.value() / 100.0

        dlg = SplitDialog(
            item.folder,
            threshold=threshold,
            parent=self,
            precomputed_result=item.result,
        )
        result = dlg.exec()

        if result == SplitDialog.DialogCode.Accepted:
            # 拆分完成 — 标记为已拆分（分组数设为 0 表示已处理）
            self._scan_results[row] = BatchScanItem(
                folder=item.folder,
                image_count=item.image_count,
                group_count=0,
                result=item.result,
                error=None,
            )
            # 更新状态列为"已拆分"
            status_cell = self._table.item(row, 0)
            status_cell.setText("OK")
            status_cell.setToolTip("已拆分")
            status_cell.setForeground(Qt.GlobalColor.darkGreen)
            self._table.item(row, 3).setText("已拆分")
            lbl = QLabel("已完成")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: green;")
            self._table.setCellWidget(row, 4, lbl)
            self._update_summary()
            self.status_message.emit(f"已拆分: {item.folder.name}")

    # ── 辅助方法 ───────────────────────────────────────────────

    @staticmethod
    def _find_image_folders(base_dir: Path) -> list[Path]:
        return find_image_folders(base_dir)

    def set_download_dir(self, path: Path) -> None:
        """外部设置下载目录（如同步采集页路径）。"""
        self._download_dir = path
        self._dir_label.setText(str(path))

    def cleanup(self) -> None:
        """应用关闭时调用，取消后台 Worker。"""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
