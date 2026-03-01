"""主图选择 + 白底图转换对话框"""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..config import ORIGINALS_SUBFOLDER, SQUARE_JPEG_QUALITY, THUMBNAIL_SIZE
from ..image_processor import (
    save_image,
    is_white_background,
    list_images,
    remove_background_to_white,
    reorder_main_image,
)
from .base_worker import BaseWorker
from .split_dialog import FlowLayout


# ── ClickableThumbnailWidget ─────────────────────────────────


class ClickableThumbnailWidget(QLabel):
    """可点击选中的缩略图标签。"""

    clicked = Signal(object)  # 发出自身

    def __init__(self, image_path: Path, thumb_size: int = THUMBNAIL_SIZE) -> None:
        super().__init__()
        self.image_path = image_path
        self._thumb_size = thumb_size
        self._selected = False

        pixmap = QPixmap(str(image_path))
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                thumb_size, thumb_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.setPixmap(pixmap)
        self.setFixedSize(thumb_size, thumb_size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._apply_style()
        self.setToolTip(image_path.name)

    def _apply_style(self) -> None:
        if self._selected:
            self.setStyleSheet(
                "border: 3px solid #2196F3; background: white; padding: 1px;"
            )
        else:
            self.setStyleSheet(
                "border: 1px solid #ccc; background: white; padding: 2px;"
            )

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._apply_style()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self)
        super().mousePressEvent(event)


# ── RemoveBgWorker ───────────────────────────────────────────


class RemoveBgWorker(BaseWorker):
    """后台去除背景。"""

    finished_ok = Signal(object)  # PIL Image

    def __init__(self, image_path: Path) -> None:
        super().__init__()
        self._path = image_path

    def _run(self) -> None:
        result = remove_background_to_white(self._path)
        self.finished_ok.emit(result)


# ── CompareDialog ────────────────────────────────────────────


class CompareDialog(QDialog):
    """原图 vs 白底图对比对话框。"""

    def __init__(
        self,
        image_path: Path,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinMaxButtonsHint
        )
        self._image_path = image_path
        self._white_bg_image = None  # PIL Image
        self._worker: RemoveBgWorker | None = None
        self._accepted = False

        self.setWindowTitle("白底图转换")
        self.setMinimumSize(700, 450)
        self.resize(800, 500)

        self._build_ui()
        self._start_worker()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # 对比区域
        compare_layout = QHBoxLayout()

        # 左：原图
        left_group = QGroupBox("原图")
        left_layout = QVBoxLayout(left_group)
        self._left_label = QLabel()
        self._left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(str(self._image_path))
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                350, 350,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._left_label.setPixmap(pixmap)
        left_layout.addWidget(self._left_label)
        compare_layout.addWidget(left_group)

        # 右：白底图
        right_group = QGroupBox("白底图")
        right_layout = QVBoxLayout(right_group)
        self._right_label = QLabel("正在处理...")
        self._right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self._right_label)
        compare_layout.addWidget(right_group)

        main_layout.addLayout(compare_layout, 1)

        # 按钮栏
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._btn_reject = QPushButton("不使用白底图")
        self._btn_reject.clicked.connect(self.reject)
        self._btn_accept = QPushButton("使用白底图")
        self._btn_accept.setEnabled(False)
        self._btn_accept.clicked.connect(self._on_accept)
        btn_layout.addWidget(self._btn_reject)
        btn_layout.addWidget(self._btn_accept)
        main_layout.addLayout(btn_layout)

    def _start_worker(self) -> None:
        self._worker = RemoveBgWorker(self._image_path)
        self._worker.finished_ok.connect(self._on_result)
        self._worker.finished_err.connect(self._on_error)
        self._worker.start()

    @Slot(object)
    def _on_result(self, pil_image) -> None:
        self._white_bg_image = pil_image
        # PIL → QPixmap
        from io import BytesIO
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                350, 350,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._right_label.setPixmap(pixmap)
        self._btn_accept.setEnabled(True)

    @Slot(str)
    def _on_error(self, error: str) -> None:
        self._right_label.setText(f"处理失败:\n{error}")

    @Slot()
    def _on_accept(self) -> None:
        if self._white_bg_image is None:
            return
        # 备份原图到 originals/ 子目录
        originals_dir = self._image_path.parent / ORIGINALS_SUBFOLDER
        originals_dir.mkdir(exist_ok=True)
        backup_path = originals_dir / self._image_path.name
        shutil.copy2(str(self._image_path), str(backup_path))

        # 保存白底图覆盖原文件
        save_image(self._white_bg_image, self._image_path, SQUARE_JPEG_QUALITY)

        self._accepted = True
        self.accept()

    @property
    def was_accepted(self) -> bool:
        return self._accepted

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.wait(5000)
        super().closeEvent(event)


# ── MainImageDialog ──────────────────────────────────────────


class MainImageDialog(QDialog):
    """主图选择对话框 — 缩略图网格，点击选中主图。"""

    def __init__(
        self,
        folder: Path,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinMaxButtonsHint
        )
        self._folder = folder
        self._selected: ClickableThumbnailWidget | None = None
        self._thumbnails: list[ClickableThumbnailWidget] = []

        self.setWindowTitle(f"选择主图 — {folder.name}")
        self.setMinimumSize(600, 400)
        self.resize(700, 500)

        self._build_ui()
        self._load_images()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # 提示
        hint = QLabel("点击选择主图（排到第一位），然后确认")
        main_layout.addWidget(hint)

        # 缩略图滚动区域
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll_content = QWidget()
        self._flow_layout = FlowLayout(self._scroll_content, spacing=6)
        self._scroll.setWidget(self._scroll_content)
        main_layout.addWidget(self._scroll, 1)

        # 按钮栏
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        self._btn_confirm = QPushButton("确认选择")
        self._btn_confirm.setEnabled(False)
        self._btn_confirm.clicked.connect(self._on_confirm)
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(self._btn_confirm)
        main_layout.addLayout(btn_layout)

    def _load_images(self) -> None:
        images = list_images(self._folder)
        for img_path in images:
            thumb = ClickableThumbnailWidget(img_path)
            thumb.clicked.connect(self._on_thumb_clicked)
            self._flow_layout.addWidget(thumb)
            self._thumbnails.append(thumb)

    @Slot(object)
    def _on_thumb_clicked(self, thumb: ClickableThumbnailWidget) -> None:
        # 取消上一个选中
        if self._selected is not None:
            self._selected.set_selected(False)
        # 选中当前
        thumb.set_selected(True)
        self._selected = thumb
        self._btn_confirm.setEnabled(True)

    @Slot()
    def _on_confirm(self) -> None:
        if self._selected is None:
            return

        main_path = self._selected.image_path

        # 检测是否白底
        if is_white_background(main_path):
            # 已是白底，直接重排序
            reorder_main_image(self._folder, main_path)
            QMessageBox.information(self, "完成", "已设为主图并重排序")
            self.accept()
            return

        # 非白底，提示是否转换
        reply = QMessageBox.question(
            self,
            "非白底主图",
            "选中的主图背景不是纯白色。\n是否转换为白底图？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # 打开对比对话框
            dlg = CompareDialog(main_path, parent=self)
            dlg.exec()
            # 无论接受还是拒绝白底图，都重排序
            # 如果接受了白底图，main_path 已被覆盖为白底图
            # 但文件可能已被保存到同一路径，所以 main_path 仍有效
            reorder_main_image(self._folder, main_path)
            if dlg.was_accepted:
                QMessageBox.information(self, "完成", "已转换为白底图并设为主图")
            else:
                QMessageBox.information(self, "完成", "已设为主图（保留原图）")
            self.accept()
        else:
            # 不转换，仅重排序
            reorder_main_image(self._folder, main_path)
            QMessageBox.information(self, "完成", "已设为主图并重排序")
            self.accept()
