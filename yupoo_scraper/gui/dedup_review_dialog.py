"""查重审核对话框 — 左右对比新产品 vs 已有产品，可选删除任意一侧"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Slot

logger = logging.getLogger(__name__)
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..config import THUMBNAIL_SIZE
from ..ml.deduplicator import DedupMatch, DedupScanItem
from ..image_processor import list_images
from .split_dialog import FlowLayout, ThumbnailWidget


# done() 返回码
DELETE_NEW = 2          # 删除新产品（左侧）
DELETE_EXISTING = 3     # 删除已有产品（右侧）
# QDialog.Accepted = 保留两者，新产品入库
# QDialog.Rejected = 取消


class DedupReviewDialog(QDialog):
    """查重审核弹窗：左侧新产品，右侧已有产品，可选删除任意一侧。"""

    def __init__(
        self,
        scan_item: DedupScanItem,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinMaxButtonsHint
        )
        self._scan_item = scan_item
        self._matches = scan_item.all_matches
        self._current_match_idx = 0

        best_sim = scan_item.best_match.similarity if scan_item.best_match else 0
        self.setWindowTitle(f"查重审核 — 相似度 {best_sim:.2f}")
        self.setMinimumSize(900, 600)
        self.resize(1000, 700)

        self._build_ui()
        self._show_match(0)

    @property
    def selected_match(self) -> DedupMatch | None:
        """返回当前显示的匹配项（用于删除已有产品时识别目标）。"""
        if 0 <= self._current_match_idx < len(self._matches):
            return self._matches[self._current_match_idx]
        return None

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # ── 左右对比区 ──────────────────────────────────────────
        compare_layout = QHBoxLayout()

        # 左侧：新采集产品
        left_group = QGroupBox("新采集产品")
        left_layout = QVBoxLayout(left_group)

        self._new_name_label = QLabel(f"名称: {self._scan_item.name}")
        self._new_name_label.setWordWrap(True)
        self._new_count_label = QLabel(f"图片数: {self._scan_item.image_count}")
        left_layout.addWidget(self._new_name_label)
        left_layout.addWidget(self._new_count_label)

        # 新产品缩略图
        new_scroll = QScrollArea()
        new_scroll.setWidgetResizable(True)
        new_thumb_container = QWidget()
        self._new_flow = FlowLayout(new_thumb_container, spacing=4)
        new_scroll.setWidget(new_thumb_container)
        left_layout.addWidget(new_scroll, 1)

        # 加载新产品图片
        new_images = list_images(self._scan_item.folder)
        for img_path in new_images:
            thumb = ThumbnailWidget(img_path, THUMBNAIL_SIZE)
            self._new_flow.addWidget(thumb)

        compare_layout.addWidget(left_group, 1)

        # 右侧：已有产品（内容在 _show_match 中填充）
        self._existing_group = QGroupBox("已有产品")
        existing_layout = QVBoxLayout(self._existing_group)

        self._exist_name_label = QLabel("")
        self._exist_name_label.setWordWrap(True)
        self._exist_store_label = QLabel("")
        self._exist_date_label = QLabel("")
        self._exist_sim_label = QLabel("")
        self._exist_sim_label.setStyleSheet("font-weight: bold;")
        existing_layout.addWidget(self._exist_name_label)
        existing_layout.addWidget(self._exist_store_label)
        existing_layout.addWidget(self._exist_date_label)
        existing_layout.addWidget(self._exist_sim_label)

        self._exist_scroll = QScrollArea()
        self._exist_scroll.setWidgetResizable(True)
        self._exist_thumb_container = QWidget()
        self._exist_flow = FlowLayout(self._exist_thumb_container, spacing=4)
        self._exist_scroll.setWidget(self._exist_thumb_container)
        existing_layout.addWidget(self._exist_scroll, 1)

        compare_layout.addWidget(self._existing_group, 1)
        main_layout.addLayout(compare_layout, 1)

        # ── 匹配切换行（多个匹配时显示） ────────────────────────
        nav_layout = QHBoxLayout()
        self._btn_prev = QPushButton("上一个匹配")
        self._btn_prev.clicked.connect(self._on_prev)
        self._btn_next = QPushButton("下一个匹配")
        self._btn_next.clicked.connect(self._on_next)
        self._match_label = QLabel("")
        self._match_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        nav_layout.addWidget(self._btn_prev)
        nav_layout.addWidget(self._match_label, 1)
        nav_layout.addWidget(self._btn_next)
        main_layout.addLayout(nav_layout)

        if len(self._matches) <= 1:
            self._btn_prev.setVisible(False)
            self._btn_next.setVisible(False)
            self._match_label.setVisible(False)

        # ── 操作按钮 ────────────────────────────────────────────
        btn_layout = QHBoxLayout()

        _btn_style_red = (
            "QPushButton { background-color: #e74c3c; color: white; "
            "padding: 8px 16px; font-weight: bold; }"
        )
        _btn_style_green = (
            "QPushButton { background-color: #2ecc71; color: white; "
            "padding: 8px 16px; font-weight: bold; }"
        )

        self._btn_del_new = QPushButton("删除左侧（新采集）")
        self._btn_del_new.setStyleSheet(_btn_style_red)
        self._btn_del_new.clicked.connect(self._on_delete_new)

        self._btn_del_exist = QPushButton("删除右侧（已有）")
        self._btn_del_exist.setStyleSheet(_btn_style_red)
        self._btn_del_exist.clicked.connect(self._on_delete_existing)

        self._btn_keep_both = QPushButton("不是重复 — 全部保留")
        self._btn_keep_both.setStyleSheet(_btn_style_green)
        self._btn_keep_both.clicked.connect(self.accept)

        btn_layout.addWidget(self._btn_del_new)
        btn_layout.addStretch()
        btn_layout.addWidget(self._btn_del_exist)
        btn_layout.addStretch()
        btn_layout.addWidget(self._btn_keep_both)
        main_layout.addLayout(btn_layout)

    def _show_match(self, idx: int) -> None:
        """显示第 idx 个匹配项的详情。"""
        if idx < 0 or idx >= len(self._matches):
            return

        self._current_match_idx = idx
        match = self._matches[idx]
        product = match.existing_product

        self._exist_name_label.setText(f"名称: {product.name}")
        self._exist_store_label.setText(f"店铺: {product.store}" if product.store else "")
        self._exist_date_label.setText(f"下载日期: {product.download_date}" if product.download_date else "")
        self._exist_sim_label.setText(f"相似度: {match.similarity:.4f}")

        # 更新导航
        self._match_label.setText(f"匹配 {idx + 1} / {len(self._matches)}")
        self._btn_prev.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < len(self._matches) - 1)

        # 清理旧容器（takeWidget 移除所有权后手动销毁，避免内存泄漏）
        old_widget = self._exist_scroll.takeWidget()
        if old_widget is not None:
            old_widget.deleteLater()
        self._exist_thumb_container = QWidget()
        self._exist_flow = FlowLayout(self._exist_thumb_container, spacing=4)
        self._exist_scroll.setWidget(self._exist_thumb_container)

        # 加载已有产品图片（尝试修正路径：如果绝对路径不存在，尝试用当前下载目录拼接）
        exist_folder = Path(product.folder)
        if not exist_folder.exists():
            from ..config import DEFAULT_DOWNLOAD_DIR
            alt_folder = DEFAULT_DOWNLOAD_DIR / exist_folder.name
            if alt_folder.exists():
                exist_folder = alt_folder
        logger.info("加载已有产品图片: folder=%s, exists=%s", exist_folder, exist_folder.exists())
        if exist_folder.exists():
            exist_images = list_images(exist_folder)
            logger.info("已有产品图片数: %d", len(exist_images))
            if exist_images:
                for img_path in exist_images:
                    thumb = ThumbnailWidget(img_path, THUMBNAIL_SIZE)
                    self._exist_flow.addWidget(thumb)
            else:
                lbl = QLabel(f"文件夹内无图片:\n{exist_folder}")
                lbl.setWordWrap(True)
                lbl.setStyleSheet("color: #666; padding: 8px;")
                self._exist_flow.addWidget(lbl)
        else:
            lbl = QLabel(f"文件夹不存在:\n{exist_folder}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: red; padding: 8px;")
            self._exist_flow.addWidget(lbl)

    @Slot()
    def _on_prev(self) -> None:
        if self._current_match_idx > 0:
            self._show_match(self._current_match_idx - 1)

    @Slot()
    def _on_next(self) -> None:
        if self._current_match_idx < len(self._matches) - 1:
            self._show_match(self._current_match_idx + 1)

    @Slot()
    def _on_delete_new(self) -> None:
        """删除新采集产品（左侧）。"""
        self.done(DELETE_NEW)

    @Slot()
    def _on_delete_existing(self) -> None:
        """删除已有产品（右侧当前显示的）。"""
        self.done(DELETE_EXISTING)
