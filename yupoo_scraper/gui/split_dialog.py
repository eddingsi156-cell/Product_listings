"""产品拆分审核对话框 — 聚类预览、灵敏度调节、拖拽修正"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6.QtCore import (
    QByteArray,
    QEvent,
    QMimeData,
    QPoint,
    QRect,
    QSize,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QDrag, QImageReader, QKeyEvent, QMouseEvent, QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRubberBand,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from ..config import (
    CLUSTER_THRESHOLD_DEFAULT,
    CLUSTER_THRESHOLD_MAX,
    CLUSTER_THRESHOLD_MIN,
    COMBINED_DIM,
    THUMBNAIL_SIZE,
)
from ..ml.splitter import (
    SplitGroup,
    SplitResult,
    apply_split,
    extract_and_split,
    recluster,
)

# 自定义 MIME 类型
MIME_IMAGE_PATH = "application/x-product-split-image"


# ── FlowLayout ─────────────────────────────────────────────────

class FlowLayout(QLayout):
    """自动换行的流式布局（Qt 不内置此布局）。"""

    def __init__(self, parent: QWidget | None = None, spacing: int = 4) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._spacing = spacing

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QLayoutItem | None:
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        m = self.contentsMargins()
        effective = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x = effective.x()
        y = effective.y()
        line_height = 0

        for item in self._items:
            sz = item.sizeHint()
            next_x = x + sz.width() + self._spacing
            if next_x - self._spacing > effective.right() and line_height > 0:
                x = effective.x()
                y += line_height + self._spacing
                next_x = x + sz.width() + self._spacing
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), sz))

            x = next_x
            line_height = max(line_height, sz.height())

        return y + line_height - rect.y() + m.bottom()


# ── ThumbnailWidget ────────────────────────────────────────────

class ThumbnailWidget(QLabel):
    """可拖拽的缩略图标签，支持选中状态。"""

    clicked = Signal(object, object)  # (self, QMouseEvent)
    drag_started = Signal(object)  # (self,)

    _STYLE_NORMAL = "border: 1px solid #ccc; background: white; padding: 2px;"
    _STYLE_SELECTED = "border: 3px solid #3daee9; background: #d4edfc; padding: 0px;"

    def __init__(self, image_path: Path, thumb_size: int = THUMBNAIL_SIZE) -> None:
        super().__init__()
        self.image_path = image_path
        self._thumb_size = thumb_size
        self._selected = False

        reader = QImageReader(str(image_path))
        reader.setAutoTransform(True)
        orig_size = reader.size()
        if orig_size.isValid():
            orig_size.scale(thumb_size, thumb_size, Qt.AspectRatioMode.KeepAspectRatio)
            reader.setScaledSize(orig_size)
        pixmap = QPixmap.fromImageReader(reader)
        self.setPixmap(pixmap)
        self.setFixedSize(thumb_size, thumb_size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(self._STYLE_NORMAL)
        self.setToolTip(image_path.name)

        # 批量拖拽时由 SplitDialog 设置
        self._batch_drag_paths: list[str] | None = None

    @property
    def selected(self) -> bool:
        return self._selected

    @selected.setter
    def selected(self, value: bool) -> None:
        if self._selected == value:
            return
        self._selected = value
        self.setStyleSheet(self._STYLE_SELECTED if value else self._STYLE_NORMAL)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
            self.clicked.emit(self, event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if not hasattr(self, "_drag_start"):
            return
        dist = (event.position().toPoint() - self._drag_start).manhattanLength()
        if dist < QApplication_startDragDistance():
            return

        self.drag_started.emit(self)

        drag = QDrag(self)
        mime = QMimeData()

        # 如果有批量路径（由 SplitDialog 在 drag_started 中设置），用换行分隔
        if self._batch_drag_paths:
            payload = "\n".join(self._batch_drag_paths)
        else:
            payload = str(self.image_path)

        mime.setData(MIME_IMAGE_PATH, QByteArray(payload.encode("utf-8")))
        drag.setMimeData(mime)

        # 拖拽时显示缩略图
        pm = self.pixmap()
        if pm and not pm.isNull():
            drag.setPixmap(pm.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio))
        drag.exec(Qt.DropAction.MoveAction)

        # 清除批量路径
        self._batch_drag_paths = None

    def contextMenuEvent(self, event) -> None:
        # 委托给 SplitDialog 处理
        dialog = self._find_split_dialog()
        if dialog is not None:
            dialog._on_thumb_context_menu(self, event.globalPos())
            event.accept()
        else:
            super().contextMenuEvent(event)

    def _find_split_dialog(self) -> "SplitDialog | None":
        p = self.parent()
        while p is not None:
            if isinstance(p, SplitDialog):
                return p
            p = p.parent()
        return None


def QApplication_startDragDistance() -> int:
    from PySide6.QtWidgets import QApplication
    return QApplication.startDragDistance()


# ── GroupWidget ────────────────────────────────────────────────

class GroupWidget(QGroupBox):
    """分组容器 — 接受 drop，包含名称编辑 + 缩略图网格。"""

    image_dropped = Signal(str, int)  # (image_path_str, target_group_id)
    thumbnail_added = Signal(object)  # (ThumbnailWidget,)

    def __init__(self, group: SplitGroup, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.group_id = group.id
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 4)

        # 标题行：名称编辑 + 图片数量
        header = QHBoxLayout()
        self._name_edit = QLineEdit(group.name)
        self._name_edit.setPlaceholderText("分组名称")
        self._count_label = QLabel(f"({len(group.image_paths)} 张)")
        header.addWidget(self._name_edit, 1)
        header.addWidget(self._count_label)
        layout.addLayout(header)

        # 缩略图区域
        self._flow_container = QWidget()
        self._flow_layout = FlowLayout(self._flow_container, spacing=4)
        layout.addWidget(self._flow_container)

        # 添加缩略图
        self._thumbnails: list[ThumbnailWidget] = []
        for img_path in group.image_paths:
            thumb = ThumbnailWidget(img_path)
            self._flow_layout.addWidget(thumb)
            self._thumbnails.append(thumb)

        self.setStyleSheet(
            "GroupWidget { border: 2px solid #3daee9; border-radius: 4px; "
            "margin-top: 8px; } "
            "GroupWidget::title { padding: 0 4px; }"
        )

    @property
    def group_name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def image_paths(self) -> list[Path]:
        return [t.image_path for t in self._thumbnails]

    def thumbnails(self) -> list[ThumbnailWidget]:
        return list(self._thumbnails)

    def add_thumbnail(self, image_path: Path) -> ThumbnailWidget:
        thumb = ThumbnailWidget(image_path)
        self._flow_layout.addWidget(thumb)
        self._thumbnails.append(thumb)
        self._count_label.setText(f"({len(self._thumbnails)} 张)")
        thumb.show()
        self.thumbnail_added.emit(thumb)
        return thumb

    def remove_thumbnail(self, image_path: Path) -> bool:
        for i, t in enumerate(self._thumbnails):
            if t.image_path == image_path:
                # 遍历 FlowLayout 找到对应 widget 的索引再移除
                for idx in range(self._flow_layout.count()):
                    item = self._flow_layout.itemAt(idx)
                    if item and item.widget() is t:
                        self._flow_layout.takeAt(idx)
                        break
                t.setParent(None)
                t.deleteLater()
                self._thumbnails.pop(i)
                self._count_label.setText(f"({len(self._thumbnails)} 张)")
                return True
        return False

    def remove_thumbnail_widget(self, thumb: ThumbnailWidget) -> bool:
        """直接按 widget 引用移除。"""
        if thumb not in self._thumbnails:
            return False
        for idx in range(self._flow_layout.count()):
            item = self._flow_layout.itemAt(idx)
            if item and item.widget() is thumb:
                self._flow_layout.takeAt(idx)
                break
        self._thumbnails.remove(thumb)
        thumb.setParent(None)
        thumb.deleteLater()
        self._count_label.setText(f"({len(self._thumbnails)} 张)")
        return True

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(MIME_IMAGE_PATH):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        data = event.mimeData().data(MIME_IMAGE_PATH)
        payload = bytes(data).decode("utf-8")
        # 支持多路径（换行分隔）
        for path_str in payload.split("\n"):
            path_str = path_str.strip()
            if path_str:
                self.image_dropped.emit(path_str, self.group_id)
        event.acceptProposedAction()


# ── 工作线程 ───────────────────────────────────────────────────

from .base_worker import BaseWorker


class SplitWorker(BaseWorker):
    """后台执行特征提取+聚类。"""

    status = Signal(str)
    progress = Signal(int, int)  # (current, total)
    finished_ok = Signal(object)  # SplitResult

    def __init__(self, folder: Path, threshold: float) -> None:
        super().__init__()
        self._folder = folder
        self._threshold = threshold

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
        slider_layout.addWidget(QLabel("灵敏度（少分组 ← → 多分组）："))
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
            t.selected = False
        self._selected_thumbs.clear()
        self._update_selection_label()

    def _set_selection(self, thumbs: list[ThumbnailWidget]) -> None:
        self._clear_selection()
        for t in thumbs:
            t.selected = True
            self._selected_thumbs.append(t)
        self._update_selection_label()

    def _toggle_selection(self, thumb: ThumbnailWidget) -> None:
        if thumb in self._selected_thumbs:
            thumb.selected = False
            self._selected_thumbs.remove(thumb)
        else:
            thumb.selected = True
            self._selected_thumbs.append(thumb)
        self._update_selection_label()

    def _update_selection_label(self) -> None:
        n = len(self._selected_thumbs)
        self._selection_label.setText(f"已选择 {n} 张" if n > 0 else "")

    def _on_thumb_clicked(self, thumb: ThumbnailWidget, event: QMouseEvent) -> None:
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
        if thumb in self._selected_thumbs and len(self._selected_thumbs) > 1:
            thumb._batch_drag_paths = [
                str(t.image_path) for t in self._selected_thumbs
            ]
        else:
            thumb._batch_drag_paths = None

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
        """防抖后实际执行重聚类。"""
        if self._result is None:
            return
        self.setCursor(Qt.CursorShape.WaitCursor)
        self._result = recluster(self._result, self._threshold)
        self._rebuild_groups()
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
        super().closeEvent(event)
