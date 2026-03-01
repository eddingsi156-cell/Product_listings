"""特征库浏览对话框 — 查看、搜索、预览、删除已入库产品"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import THUMBNAIL_SIZE
from ..ml.deduplicator import Deduplicator, ProductRecord
from ..image_processor import list_images
from .split_dialog import FlowLayout


class FeatureStoreBrowserDialog(QDialog):
    """特征库浏览对话框：列出所有已入库产品，支持搜索、缩略图预览和删除。"""

    def __init__(self, deduplicator: Deduplicator, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinMaxButtonsHint
        )
        self._dedup = deduplicator
        self._products: list[ProductRecord] = []
        self._deleted_ids: set[int] = set()

        self.setWindowTitle("浏览特征库")
        self.resize(800, 600)
        self._build_ui()
        self._load_products()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── 搜索栏 ────────────────────────────────────────────
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("搜索:"))
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("输入关键词筛选...")
        self._search_edit.textChanged.connect(self._on_search)
        search_layout.addWidget(self._search_edit, 1)
        self._count_label = QLabel("共 0 个产品")
        search_layout.addWidget(self._count_label)
        layout.addLayout(search_layout)

        # ── 产品表格 ──────────────────────────────────────────
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["名称", "店铺", "文件夹", "下载日期", "图片数"]
        )
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        self._table.currentCellChanged.connect(self._on_row_selected)
        layout.addWidget(self._table, 1)

        # ── 缩略图预览 ────────────────────────────────────────
        self._preview_container = QWidget()
        self._preview_layout = FlowLayout(self._preview_container, spacing=4)
        self._preview_container.setLayout(self._preview_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._preview_container)
        scroll.setMinimumHeight(150)
        scroll.setMaximumHeight(200)
        layout.addWidget(scroll)

        self._preview_hint = QLabel("")
        self._preview_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._preview_hint)

        # ── 底部按钮 ──────────────────────────────────────────
        btn_layout = QHBoxLayout()
        self._btn_delete = QPushButton("删除选中产品")
        self._btn_delete.clicked.connect(self._on_delete)
        btn_layout.addWidget(self._btn_delete)
        btn_layout.addStretch()
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

    def _load_products(self) -> None:
        """从 Deduplicator 加载所有产品到表格。"""
        self._products = self._dedup.get_all_products()
        self._populate_table(self._products)

    def _populate_table(self, products: list[ProductRecord]) -> None:
        """填充表格。"""
        self._table.setRowCount(0)
        self._table.setRowCount(len(products))
        for row, p in enumerate(products):
            self._table.setItem(row, 0, QTableWidgetItem(p.name))
            self._table.setItem(row, 1, QTableWidgetItem(p.store))
            self._table.setItem(row, 2, QTableWidgetItem(p.folder))
            self._table.setItem(row, 3, QTableWidgetItem(p.download_date))
            count_item = QTableWidgetItem(str(p.image_count))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 4, count_item)
        self._count_label.setText(f"共 {len(products)} 个产品")

    @Slot(str)
    def _on_search(self, text: str) -> None:
        """按关键词筛选表格行。"""
        keyword = text.strip().lower()
        visible = 0
        for row in range(self._table.rowCount()):
            name_item = self._table.item(row, 0)
            folder_item = self._table.item(row, 2)
            store_item = self._table.item(row, 1)
            match = (
                not keyword
                or keyword in (name_item.text().lower() if name_item else "")
                or keyword in (folder_item.text().lower() if folder_item else "")
                or keyword in (store_item.text().lower() if store_item else "")
            )
            self._table.setRowHidden(row, not match)
            if match:
                visible += 1
        self._count_label.setText(f"共 {visible} 个产品")

    @Slot(int, int, int, int)
    def _on_row_selected(self, row: int, _col: int, _prev_row: int, _prev_col: int) -> None:
        """点击行时预览该产品的缩略图。"""
        self._clear_preview()
        if row < 0 or row >= len(self._products):
            return

        product = self._products[row]
        folder = Path(product.folder)

        if not folder.exists():
            self._preview_hint.setText("文件夹不存在")
            self._preview_hint.setStyleSheet("color: red;")
            return

        images = list_images(folder)
        if not images:
            self._preview_hint.setText("文件夹中无图片")
            self._preview_hint.setStyleSheet("color: gray;")
            return

        self._preview_hint.setText("")
        self._preview_hint.setStyleSheet("")
        for img_path in images[:20]:  # 最多显示 20 张
            thumb = self._make_thumbnail(img_path)
            self._preview_layout.addWidget(thumb)

        if len(images) > 20:
            more = QLabel(f"+{len(images) - 20}")
            more.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
            more.setAlignment(Qt.AlignmentFlag.AlignCenter)
            more.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0; color: gray;")
            self._preview_layout.addWidget(more)

    def _make_thumbnail(self, image_path: Path) -> QLabel:
        """创建缩略图 QLabel（读取时预缩放，避免加载全尺寸图片）。"""
        label = QLabel()
        reader = QImageReader(str(image_path))
        reader.setAutoTransform(True)
        orig_size = reader.size()
        if orig_size.isValid():
            orig_size.scale(THUMBNAIL_SIZE, THUMBNAIL_SIZE, Qt.AspectRatioMode.KeepAspectRatio)
            reader.setScaledSize(orig_size)
        pixmap = QPixmap.fromImageReader(reader)
        label.setPixmap(pixmap)
        label.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("border: 1px solid #ccc; background: white; padding: 2px;")
        label.setToolTip(image_path.name)
        return label

    def _clear_preview(self) -> None:
        """清除缩略图预览区域。"""
        while self._preview_layout.count():
            item = self._preview_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._preview_hint.setText("")
        self._preview_hint.setStyleSheet("")

    @Slot()
    def _on_delete(self) -> None:
        """删除选中的产品。"""
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()})
        if not rows:
            QMessageBox.information(self, "提示", "请先选择要删除的产品")
            return

        names = [self._products[r].name for r in rows if r < len(self._products)]
        msg = f"确定要从特征库中删除以下 {len(names)} 个产品？\n\n"
        msg += "\n".join(f"  - {n}" for n in names[:10])
        if len(names) > 10:
            msg += f"\n  ... 及另外 {len(names) - 10} 个"
        msg += "\n\n注意：只会删除特征库记录，不会删除原始文件。"

        ret = QMessageBox.question(
            self, "确认删除", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return

        # 按倒序删除，避免索引偏移
        for r in reversed(rows):
            if r >= len(self._products):
                continue
            product = self._products[r]
            self._dedup.remove_product(product.id)
            self._deleted_ids.add(product.id)

        # 持久化 FAISS 索引
        if self._deleted_ids:
            self._dedup.save_index()

        # 重新加载
        self._clear_preview()
        self._load_products()
        self._on_search(self._search_edit.text())

    @property
    def has_deletions(self) -> bool:
        """是否发生了删除操作。"""
        return len(self._deleted_ids) > 0
