"""可拖拽的缩略图标签"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QByteArray, QEvent, QMimeData, QPoint, Qt, Signal
from PySide6.QtGui import QDrag, QImageReader, QMouseEvent, QPixmap
from PySide6.QtWidgets import QLabel

from ..config import THUMBNAIL_SIZE

if TYPE_CHECKING:
    from .split_dialog import SplitDialog

# 自定义 MIME 类型
MIME_IMAGE_PATH = "application/x-product-split-image"


def QApplication_startDragDistance() -> int:
    from PySide6.QtWidgets import QApplication
    return QApplication.startDragDistance()


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
        try:
            # 检查 widget 是否仍然有效
            if hasattr(self, 'parent') and self.parent() is not None:
                self.setStyleSheet(self._STYLE_SELECTED if value else self._STYLE_NORMAL)
        except RuntimeError:
            # 忽略已删除的 widget
            pass

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

    def _find_split_dialog(self) -> SplitDialog | None:
        from .split_dialog import SplitDialog
        p = self.parent()
        while p is not None:
            if isinstance(p, SplitDialog):
                return p
            p = p.parent()
        return None
