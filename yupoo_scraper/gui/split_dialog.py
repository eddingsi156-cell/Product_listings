"""产品拆分审核对话框 — 聚类预览、灵敏度调节、拖拽修正"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6.QtCore import (
    QEvent,
    QPoint,
    QRect,
    QSize,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QKeyEvent, QMouseEvent, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRubberBand,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    CLUSTER_THRESHOLD_DEFAULT,
    CLUSTER_THRESHOLD_MAX,
    CLUSTER_THRESHOLD_MIN,
    COMBINED_DIM,
)
from ..ml.splitter import (
    SplitGroup,
    SplitResult,
    apply_split,
    extract_and_split,
    recluster,
)

from .base_worker import BaseWorker
from .flow_layout import FlowLayout
from .group_widget import GroupWidget
from .thumbnail_widget import ThumbnailWidget


# ── 工作线程 ───────────────────────────────────────────────────



class SplitWorker(BaseWorker):
    """后台执行特征提取+聚类。"""

    status = Signal(str)
    progress = Signal(int, int)  # (current, total)
    finished_ok = Signal(object)  # SplitResult

    def __init__(self, folder: Path, threshold: float, force: bool = False) -> None:
        super().__init__()
        self._folder = folder
        self._threshold = threshold
        self._force = force

    def _run(self) -> None:
        result = extract_and_split(
            self._folder,
            threshold=self._threshold,
            on_status=lambda msg: (
                self.status.emit(msg) if not self._cancelled else None
            ),
            on_progress=lambda cur, tot: (
                self.progress.emit(cur, tot) if not self._cancelled else None
            ),
            force=self._force,
        )
        if not self._cancelled:
            self.finished_ok.emit(result)


class ApplyWorker(BaseWorker):
    """后台执行文件移动。"""

    progress = Signal(int, int)
    finished_ok = Signal(list)  # list[Path]

    def __init__(self, result: SplitResult) -> None:
        super().__init__()
        self._result = result

    def _run(self) -> None:
        folders = apply_split(
            self._result,
            on_progress=lambda cur, tot: self.progress.emit(cur, tot),
        )
        self.finished_ok.emit(folders)


class ReclusterWorker(BaseWorker):
    """后台执行重聚类（大数据集）。"""

    finished_ok = Signal(object)

    def __init__(self, result: SplitResult, threshold: float) -> None:
        super().__init__()
        self._result = result
        self._threshold = threshold

    def _run(self) -> None:
        new_result = recluster(self._result, self._threshold)
        if not self._cancelled:
            self.finished_ok.emit(new_result)


# ── SplitDialog ────────────────────────────────────────────────

class SplitDialog(QDialog):
    """产品拆分审核对话框。"""

    def __init__(
        self,
        folder: Path,
        threshold: float = CLUSTER_THRESHOLD_DEFAULT,
        parent: QWidget | None = None,
        precomputed_result: SplitResult | None = None,
    ):
        super().__init__(parent)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinMaxButtonsHint
        )
        self._folder = folder
        self._threshold = threshold
        self._result: SplitResult | None = None
        self._worker: SplitWorker | None = None
        self._apply_worker: ApplyWorker | None = None
        self._recluster_worker: BaseWorker | None = None
        self._group_widgets: list[GroupWidget] = []
        self._recluster_timer = QTimer(self)
        self._recluster_timer.setSingleShot(True)
        self._recluster_timer.setInterval(300)
        self._recluster_timer.timeout.connect(self._do_recluster)

        # ── 选择管理 ──
        self._selected_thumbs: list[ThumbnailWidget] = []
        self._last_clicked_thumb: ThumbnailWidget | None = None
        self._rubber_band: QRubberBand | None = None
        self._rubber_band_origin: QPoint | None = None

        self.setWindowTitle(f"产品拆分 — {folder.name}")
        self.setMinimumSize(800, 600)
        self.resize(900, 700)

        self._build_ui()
        self._setup_shortcuts()

        if precomputed_result is not None:
            self._on_worker_done(precomputed_result)
        else:
            self._start_worker()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # ── 状态 / 进度区 ──────────────────────────────────────
        status_layout = QHBoxLayout()
        self._status_label = QLabel("正在准备...")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # indeterminate
        status_layout.addWidget(self._status_label, 1)
        status_layout.addWidget(self._progress_bar, 1)
        main_layout.addLayout(status_layout)

        # ── 灵敏度滑块 ────────────────────────────────────────
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("灵敏度（多分组 ← → 少分组）："))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(int(CLUSTER_THRESHOLD_MIN * 100))
        self._slider.setMaximum(int(CLUSTER_THRESHOLD_MAX * 100))
        self._slider.setValue(int(self._threshold * 100))
        self._slider.setTickInterval(5)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._threshold_label = QLabel(f"{self._threshold:.2f}")
        slider_layout.addWidget(self._slider, 1)
        slider_layout.addWidget(self._threshold_label)
        main_layout.addLayout(slider_layout)

        # ── 分组滚动区 ────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll_content = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.addStretch()
        self._scroll.setWidget(self._scroll_content)
        main_layout.addWidget(self._scroll, 1)

        # 安装事件过滤器用于框选
        self._scroll_content.installEventFilter(self)

        # ── 按钮栏 ────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        self._btn_new_group = QPushButton("新建分组")
        self._btn_new_group.clicked.connect(self._on_new_group)
        self._btn_new_group.setEnabled(False)
        btn_layout.addWidget(self._btn_new_group)
        btn_layout.addStretch()

        self._selection_label = QLabel("")
        btn_layout.addWidget(self._selection_label)
        btn_layout.addStretch()

        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_confirm = QPushButton("确认拆分")
        self._btn_confirm.clicked.connect(self._on_confirm)
        self._btn_confirm.setEnabled(False)
        btn_layout.addWidget(self._btn_cancel)
        btn_layout.addWidget(self._btn_confirm)
        main_layout.addLayout(btn_layout)

    def _setup_shortcuts(self) -> None:
        # Ctrl+A: 全选当前分组
        sc_select_all = QShortcut(QKeySequence.StandardKey.SelectAll, self)
        sc_select_all.activated.connect(self._on_select_all)

        # Delete: 删除选中图片
        sc_delete = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        sc_delete.activated.connect(self._on_delete_selected)

        # Escape: 清除选择
        sc_escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        sc_escape.activated.connect(self._clear_selection)

    # ── 选择管理 ───────────────────────────────────────────────

    def _all_thumbs_ordered(self) -> list[ThumbnailWidget]:
        """按分组顺序返回所有缩略图的线性列表。"""
        result: list[ThumbnailWidget] = []
        for gw in self._group_widgets:
            result.extend(gw.thumbnails())
        return result

    def _find_group_for_thumb(self, thumb: ThumbnailWidget) -> GroupWidget | None:
        for gw in self._group_widgets:
            if thumb in gw._thumbnails:
                return gw
        return None

    def _clear_selection(self) -> None:
        for t in self._selected_thumbs:
            # 检查 widget 是否仍然有效（C++ 对象未被删除）
            if t and hasattr(t, 'parent') and t.parent() is not None:
                try:
                    t.selected = False
                except RuntimeError:
                    pass
        self._selected_thumbs.clear()
        self._update_selection_label()

    def _set_selection(self, thumbs: list[ThumbnailWidget]) -> None:
        self._clear_selection()
        for t in thumbs:
            # 检查 widget 是否仍然有效
            if t and hasattr(t, 'parent') and t.parent() is not None:
                try:
                    t.selected = True
                    self._selected_thumbs.append(t)
                except RuntimeError:
                    # 忽略已删除的 widget
                    pass
        self._update_selection_label()

    def _toggle_selection(self, thumb: ThumbnailWidget) -> None:
        # 检查 widget 是否仍然有效
        if not (thumb and hasattr(thumb, 'parent') and thumb.parent() is not None):
            return
        try:
            if thumb in self._selected_thumbs:
                thumb.selected = False
                self._selected_thumbs.remove(thumb)
            else:
                thumb.selected = True
                self._selected_thumbs.append(thumb)
            self._update_selection_label()
        except RuntimeError:
            # 忽略已删除的 widget
            if thumb in self._selected_thumbs:
                self._selected_thumbs.remove(thumb)
                self._update_selection_label()

    def _update_selection_label(self) -> None:
        n = len(self._selected_thumbs)
        self._selection_label.setText(f"已选择 {n} 张" if n > 0 else "")

    def _on_thumb_clicked(self, thumb: ThumbnailWidget, event: QMouseEvent) -> None:
        # 检查 widget 是否仍然有效
        if not (thumb and hasattr(thumb, 'parent') and thumb.parent() is not None):
            return
        modifiers = event.modifiers()

        if modifiers & Qt.KeyboardModifier.ControlModifier:
            # Ctrl+Click: 切换
            self._toggle_selection(thumb)
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            # Shift+Click: 范围选择
            if self._last_clicked_thumb is not None:
                all_thumbs = self._all_thumbs_ordered()
                try:
                    idx_a = all_thumbs.index(self._last_clicked_thumb)
                    idx_b = all_thumbs.index(thumb)
                except ValueError:
                    self._set_selection([thumb])
                else:
                    lo, hi = sorted((idx_a, idx_b))
                    self._set_selection(all_thumbs[lo : hi + 1])
            else:
                self._set_selection([thumb])
        else:
            # 普通点击: 单选
            self._set_selection([thumb])

        self._last_clicked_thumb = thumb

    def _on_thumb_drag_started(self, thumb: ThumbnailWidget) -> None:
        """拖拽开始时，如果该缩略图在选中集合中，则批量拖拽所有选中图片。"""
        # 检查 widget 是否仍然有效
        if not (thumb and hasattr(thumb, 'parent') and thumb.parent() is not None):
            return
        try:
            if thumb in self._selected_thumbs and len(self._selected_thumbs) > 1:
                # 过滤掉已删除的 widget
                valid_thumbs = []
                for t in self._selected_thumbs:
                    if t and hasattr(t, 'parent') and t.parent() is not None:
                        try:
                            # 尝试访问属性以检查 widget 是否有效
                            _ = t.image_path
                            valid_thumbs.append(t)
                        except RuntimeError:
                            pass
                thumb._batch_drag_paths = [
                    str(t.image_path) for t in valid_thumbs
                ]
            else:
                thumb._batch_drag_paths = None
        except RuntimeError:
            # 忽略已删除的 widget
            pass

    def _connect_thumb_signals(self, thumb: ThumbnailWidget) -> None:
        thumb.clicked.connect(self._on_thumb_clicked)
        thumb.drag_started.connect(self._on_thumb_drag_started)

    def _on_thumbnail_added(self, thumb: ThumbnailWidget) -> None:
        self._connect_thumb_signals(thumb)

    @Slot()
    def _on_select_all(self) -> None:
        """Ctrl+A: 全选最后点击所在分组的所有缩略图。"""
        if self._last_clicked_thumb is not None:
            gw = self._find_group_for_thumb(self._last_clicked_thumb)
            if gw is not None:
                self._set_selection(gw.thumbnails())
                return
        # 如果没有上次点击位置，全选所有
        self._set_selection(self._all_thumbs_ordered())

    @Slot()
    def _on_delete_selected(self) -> None:
        """Delete 键 / 右键菜单: 彻底删除选中的图片文件。"""
        if not self._selected_thumbs:
            return

        n = len(self._selected_thumbs)
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除选中的 {n} 张图片吗？\n"
            "（图片文件将被永久删除，无法恢复！）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        thumbs_to_remove = list(self._selected_thumbs)
        self._clear_selection()
        self._last_clicked_thumb = None

        failed: list[str] = []
        for thumb in thumbs_to_remove:
            gw = self._find_group_for_thumb(thumb)
            if gw is not None:
                gw.remove_thumbnail_widget(thumb)
            # 删除磁盘文件
            try:
                thumb.image_path.unlink(missing_ok=True)
            except OSError as e:
                failed.append(f"{thumb.image_path.name}: {e}")

        if failed:
            QMessageBox.warning(
                self, "部分删除失败",
                "以下文件无法删除：\n" + "\n".join(failed),
            )

    # ── 框选（橡皮筋）──────────────────────────────────────────

    def eventFilter(self, obj, event: QEvent) -> bool:
        if obj is not self._scroll_content:
            return super().eventFilter(obj, event)

        if event.type() == QEvent.Type.MouseButtonPress:
            me: QMouseEvent = event  # type: ignore[assignment]
            if me.button() == Qt.MouseButton.LeftButton:
                # 检查是否点在缩略图上——如果是，不启动框选
                child = self._scroll_content.childAt(me.position().toPoint())
                if child is not None and self._is_or_contains_thumb(child):
                    return False
                # 在空白处开始框选
                self._rubber_band_origin = me.position().toPoint()
                if self._rubber_band is None:
                    self._rubber_band = QRubberBand(
                        QRubberBand.Shape.Rectangle, self._scroll_content
                    )
                self._rubber_band.setGeometry(
                    QRect(self._rubber_band_origin, QSize())
                )
                self._rubber_band.show()
                # 空白处点击 — 如果无 Ctrl 修饰则清除选择
                if not (me.modifiers() & Qt.KeyboardModifier.ControlModifier):
                    self._clear_selection()
                return True

        elif event.type() == QEvent.Type.MouseMove:
            if self._rubber_band is not None and self._rubber_band_origin is not None:
                me: QMouseEvent = event  # type: ignore[assignment]
                self._rubber_band.setGeometry(
                    QRect(self._rubber_band_origin, me.position().toPoint()).normalized()
                )
                return True

        elif event.type() == QEvent.Type.MouseButtonRelease:
            if self._rubber_band is not None and self._rubber_band_origin is not None:
                me: QMouseEvent = event  # type: ignore[assignment]
                rect = QRect(
                    self._rubber_band_origin, me.position().toPoint()
                ).normalized()
                self._rubber_band.hide()
                self._rubber_band_origin = None
                self._select_thumbs_in_rect(rect, me.modifiers())
                return True

        return super().eventFilter(obj, event)

    def _is_or_contains_thumb(self, widget: QWidget) -> bool:
        """检查 widget 是否是 ThumbnailWidget 或其祖先中包含 ThumbnailWidget。"""
        w = widget
        while w is not None and w is not self._scroll_content:
            if isinstance(w, ThumbnailWidget):
                return True
            w = w.parent()
        return False

    def _select_thumbs_in_rect(self, rect: QRect, modifiers) -> None:
        """选中与框选矩形相交的缩略图。"""
        ctrl = modifiers & Qt.KeyboardModifier.ControlModifier
        hits: list[ThumbnailWidget] = []

        for gw in self._group_widgets:
            for thumb in gw.thumbnails():
                # 将缩略图坐标映射到 scroll_content 坐标
                thumb_rect = QRect(
                    thumb.mapTo(self._scroll_content, QPoint(0, 0)),
                    thumb.size(),
                )
                if rect.intersects(thumb_rect):
                    hits.append(thumb)

        if ctrl:
            # Ctrl+框选: 追加/切换
            for t in hits:
                if t not in self._selected_thumbs:
                    t.selected = True
                    self._selected_thumbs.append(t)
        else:
            self._set_selection(hits)

        self._update_selection_label()

    # ── 右键菜单（缩略图）────────────────────────────────────────

    def _on_thumb_context_menu(self, thumb: ThumbnailWidget, global_pos: QPoint) -> None:
        # 如果右键点击的缩略图不在选中集合中，先选中它
        if thumb not in self._selected_thumbs:
            self._set_selection([thumb])
            self._last_clicked_thumb = thumb

        n = len(self._selected_thumbs)
        menu = QMenu(self)

        # ── 移动到分组 ──
        move_menu = menu.addMenu(f"移动 {n} 张图片到…")
        for gw in self._group_widgets:
            action = move_menu.addAction(gw.group_name or f"分组 {gw.group_id}")
            action.setData(gw.group_id)

        # ── 删除 ──
        menu.addSeparator()
        delete_action = menu.addAction(f"删除 {n} 张图片")

        chosen = menu.exec(global_pos)
        if chosen is None:
            return

        if chosen == delete_action:
            self._on_delete_selected()
        elif chosen.data() is not None:
            self._move_selected_to_group(chosen.data())

    def _move_selected_to_group(self, target_group_id: int) -> None:
        """将选中的缩略图移动到目标分组。"""
        target_gw: GroupWidget | None = None
        for gw in self._group_widgets:
            if gw.group_id == target_group_id:
                target_gw = gw
                break
        if target_gw is None:
            return

        thumbs_to_move = list(self._selected_thumbs)
        self._clear_selection()
        self._last_clicked_thumb = None

        for thumb in thumbs_to_move:
            src_gw = self._find_group_for_thumb(thumb)
            if src_gw is not None and src_gw is not target_gw:
                img_path = thumb.image_path
                src_gw.remove_thumbnail_widget(thumb)
                target_gw.add_thumbnail(img_path)

    # ── 工作线程控制 ────────────────────────────────────────────

    def _start_worker(self) -> None:
        self._worker = SplitWorker(self._folder, self._threshold)
        self._worker.status.connect(self._on_worker_status)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished_ok.connect(self._on_worker_done)
        self._worker.finished_err.connect(self._on_worker_error)
        self._worker.start()

    @Slot(str)
    def _on_worker_status(self, msg: str) -> None:
        self._status_label.setText(msg)

    @Slot(int, int)
    def _on_worker_progress(self, current: int, total: int) -> None:
        if self._progress_bar.maximum() == 0:
            self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(current)

    @Slot(object)
    def _on_worker_done(self, result: SplitResult) -> None:
        self._result = result
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(1)
        self._slider.setEnabled(True)
        self._btn_new_group.setEnabled(True)
        self._btn_confirm.setEnabled(True)
        self._rebuild_groups()

    @Slot(str)
    def _on_worker_error(self, error: str) -> None:
        self._status_label.setText(f"错误：{error}")
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        QMessageBox.critical(self, "拆分失败", error)

    # ── 灵敏度滑块 ─────────────────────────────────────────────

    @Slot(int)
    def _on_slider_changed(self, value: int) -> None:
        self._threshold = value / 100.0
        self._threshold_label.setText(f"{self._threshold:.2f}")
        if self._result is not None:
            self._recluster_timer.start()  # 防抖：300ms 内无新变化才执行

    def _do_recluster(self) -> None:
        """防抖后实际执行重聚类。

        <=500 张图片在主线程同步执行（<50ms）；
        >500 张图片在后台线程执行，避免冻结 GUI。
        """
        if self._result is None:
            return
        # 取消上一次后台重聚类（如果还在运行）
        if self._recluster_worker and self._recluster_worker.isRunning():
            self._recluster_worker.cancel()
            self._recluster_worker.wait(1000)
        n = len(self._result.image_paths)
        if n > 500:
            # 大数据集：后台线程
            self._slider.setEnabled(False)
            self.setCursor(Qt.CursorShape.WaitCursor)
            self._status_label.setText("正在重新聚类...")

            self._recluster_worker = ReclusterWorker(self._result, self._threshold)
            self._recluster_worker.finished_ok.connect(self._on_recluster_done)
            self._recluster_worker.finished_err.connect(self._on_recluster_error)
            self._recluster_worker.start()
        else:
            # 小数据集：主线程同步
            self.setCursor(Qt.CursorShape.WaitCursor)
            self._result = recluster(self._result, self._threshold)
            self._rebuild_groups()
            self.unsetCursor()

    @Slot(object)
    def _on_recluster_done(self, result: SplitResult) -> None:
        self._result = result
        self._rebuild_groups()
        self._slider.setEnabled(True)
        self.unsetCursor()

    @Slot(str)
    def _on_recluster_error(self, error: str) -> None:
        self._status_label.setText(f"重聚类失败：{error}")
        self._slider.setEnabled(True)
        self.unsetCursor()

    # ── 分组 UI 构建 ───────────────────────────────────────────

    def _rebuild_groups(self) -> None:
        if self._result is None:
            return

        # 清除选择状态
        self._clear_selection()
        self._last_clicked_thumb = None

        # 清除旧的分组
        for gw in self._group_widgets:
            gw.setParent(None)
            gw.deleteLater()
        self._group_widgets.clear()

        # 创建新的分组
        for group in self._result.groups:
            gw = GroupWidget(group)
            gw.image_dropped.connect(self._on_image_dropped)
            gw.thumbnail_added.connect(self._on_thumbnail_added)
            gw.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            gw.customContextMenuRequested.connect(
                lambda pos, g=gw: self._on_group_context_menu(g, pos)
            )
            # 连接已有缩略图的信号
            for thumb in gw.thumbnails():
                self._connect_thumb_signals(thumb)
            # 插入到 stretch 之前
            self._scroll_layout.insertWidget(
                self._scroll_layout.count() - 1, gw,
            )
            self._group_widgets.append(gw)

        n_groups = len(self._result.groups)
        n_images = len(self._result.image_paths)
        self._status_label.setText(
            f"共 {n_images} 张图片，分为 {n_groups} 个分组"
        )

    # ── 拖拽处理 ───────────────────────────────────────────────

    @Slot(str, int)
    def _on_image_dropped(self, path_str: str, target_group_id: int) -> None:
        image_path = Path(path_str)

        # 从源分组移除
        source_gw: GroupWidget | None = None
        for gw in self._group_widgets:
            if any(t.image_path == image_path for t in gw._thumbnails):
                source_gw = gw
                break

        if source_gw is None or source_gw.group_id == target_group_id:
            return

        # 找到目标分组
        target_gw: GroupWidget | None = None
        for gw in self._group_widgets:
            if gw.group_id == target_group_id:
                target_gw = gw
                break

        if target_gw is None:
            return

        # 清除被移动图片的选中状态
        for thumb in list(self._selected_thumbs):
            if thumb.image_path == image_path:
                thumb.selected = False
                self._selected_thumbs.remove(thumb)

        source_gw.remove_thumbnail(image_path)
        target_gw.add_thumbnail(image_path)
        self._update_selection_label()

    # ── 右键菜单（分组）──────────────────────────────────────────

    def _on_group_context_menu(self, group_widget: GroupWidget, pos: QPoint) -> None:
        menu = QMenu(self)
        delete_action = menu.addAction("删除此分组（图片移至第一组）")
        action = menu.exec(group_widget.mapToGlobal(pos))

        if action == delete_action:
            self._delete_group(group_widget)

    def _delete_group(self, gw: GroupWidget) -> None:
        if len(self._group_widgets) <= 1:
            QMessageBox.warning(self, "提示", "至少需要保留一个分组")
            return

        # 将图片移到第一个非自己的分组
        target = next(g for g in self._group_widgets if g is not gw)
        for path in list(gw.image_paths):
            target.add_thumbnail(path)

        self._scroll_layout.removeWidget(gw)
        self._group_widgets.remove(gw)
        gw.setParent(None)
        gw.deleteLater()

    # ── 新建分组 ──────────────────────────────────────────────

    @Slot()
    def _on_new_group(self) -> None:
        new_id = max(gw.group_id for gw in self._group_widgets) + 1 if self._group_widgets else 0
        group = SplitGroup(
            id=new_id,
            name=f"配色_{new_id + 1}",
            image_paths=[],
            original_indices=[],
        )
        gw = GroupWidget(group)
        gw.image_dropped.connect(self._on_image_dropped)
        gw.thumbnail_added.connect(self._on_thumbnail_added)
        gw.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        gw.customContextMenuRequested.connect(
            lambda pos, g=gw: self._on_group_context_menu(g, pos)
        )
        self._scroll_layout.insertWidget(
            self._scroll_layout.count() - 1, gw,
        )
        self._group_widgets.append(gw)

    # ── 确认拆分 ──────────────────────────────────────────────

    @Slot()
    def _on_confirm(self) -> None:
        # 校验：无空分组
        empty = [gw for gw in self._group_widgets if not gw.image_paths]
        if empty:
            QMessageBox.warning(
                self, "校验失败",
                f"存在 {len(empty)} 个空分组，请删除或拖入图片",
            )
            return

        # 校验：无重名
        names = [gw.group_name for gw in self._group_widgets]
        if len(set(names)) != len(names):
            QMessageBox.warning(self, "校验失败", "分组名称不能重复")
            return

        # 校验：名称不为空
        if any(not n for n in names):
            QMessageBox.warning(self, "校验失败", "分组名称不能为空")
            return

        # 构建最终 SplitResult（用 GUI 当前状态）
        groups: list[SplitGroup] = []
        for i, gw in enumerate(self._group_widgets):
            groups.append(SplitGroup(
                id=i,
                name=gw.group_name,
                image_paths=list(gw.image_paths),
                original_indices=[],
            ))

        final_result = SplitResult(
            album_folder=self._folder,
            groups=groups,
            image_paths=self._result.image_paths if self._result else [],
            features=self._result.features if self._result else np.empty((0, COMBINED_DIM)),
        )

        self._btn_confirm.setEnabled(False)
        self._btn_cancel.setEnabled(False)
        self._slider.setEnabled(False)
        self._status_label.setText("正在移动文件...")
        self._progress_bar.setRange(0, 0)

        self._apply_worker = ApplyWorker(final_result)
        self._apply_worker.progress.connect(self._on_apply_progress)
        self._apply_worker.finished_ok.connect(self._on_apply_done)
        self._apply_worker.finished_err.connect(self._on_apply_error)
        self._apply_worker.start()

    @Slot(int, int)
    def _on_apply_progress(self, current: int, total: int) -> None:
        if self._progress_bar.maximum() == 0:
            self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(current)

    @Slot(list)
    def _on_apply_done(self, folders: list) -> None:
        self._status_label.setText(f"拆分完成，创建了 {len(folders)} 个子文件夹")
        self.accept()

    @Slot(str)
    def _on_apply_error(self, error: str) -> None:
        self._status_label.setText("文件移动失败")
        self._btn_confirm.setEnabled(True)
        self._btn_cancel.setEnabled(True)
        QMessageBox.critical(self, "移动失败", error)

    # ── 安全关闭 ──────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._recluster_timer.stop()
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            # 断开信号防止 worker 完成后发到已销毁的槽
            try:
                self._worker.finished_ok.disconnect()
                self._worker.finished_err.disconnect()
                self._worker.status.disconnect()
                self._worker.progress.disconnect()
            except RuntimeError:
                pass  # 信号可能未连接
            self._worker.wait(3000)
        if self._apply_worker and self._apply_worker.isRunning():
            try:
                self._apply_worker.finished_ok.disconnect()
                self._apply_worker.finished_err.disconnect()
                self._apply_worker.progress.disconnect()
            except RuntimeError:
                pass
            self._apply_worker.wait(3000)
        if self._recluster_worker and self._recluster_worker.isRunning():
            self._recluster_worker.cancel()
            try:
                self._recluster_worker.finished_ok.disconnect()
                self._recluster_worker.finished_err.disconnect()
            except (RuntimeError, AttributeError):
                pass
            self._recluster_worker.wait(3000)
        super().closeEvent(event)
