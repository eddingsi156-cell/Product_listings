"""分组容器 - 接受 drop，包含名称编辑 + 缩略图网格"""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLineEdit, QLabel, QVBoxLayout, QWidget

from .flow_layout import FlowLayout
from .thumbnail_widget import ThumbnailWidget, MIME_IMAGE_PATH
from ..ml.splitter import SplitGroup


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
