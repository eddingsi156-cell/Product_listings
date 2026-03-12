"""查重标签页 — 批量扫描 + 结果表格 + 审核联动 + 批量注册"""

from __future__ import annotations

import logging
import shutil
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

from PySide6.QtCore import QModelIndex, QSortFilterProxyModel, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .. import config
from ..ml.deduplicator import (
    DedupMatch,
    DedupScanItem,
    DedupStatus,
    Deduplicator,
)
from ..config import IMAGE_EXTS, THUMBNAIL_SIZE
from ..image_processor import list_images
from ..organizer import find_image_folders
from .dedup_review_dialog import DELETE_EXISTING, DELETE_NEW, DedupReviewDialog
from .feature_store_dialog import FeatureStoreBrowserDialog
from .split_dialog import FlowLayout, ThumbnailWidget
from .models import (
    ButtonDelegate,
    ROLE_ACTION_COLOR,
    ROLE_ACTION_TEXT,
    ROLE_ACTION_TOOLTIP,
    ROLE_ACTION_TYPE,
    ROLE_RAW_DATA,
    VirtualTableModel,
)
from .widgets import StatusProgressBar


# ── 筛选枚举 ───────────────────────────────────────────────────

_FILTER_ALL = 0
_FILTER_DUPLICATE = 1
_FILTER_REVIEW = 2
_FILTER_NEW = 3
_FILTER_ERROR = 4

# ── 行颜色 ─────────────────────────────────────────────────────

_COLOR_DUPLICATE = QColor(255, 200, 200)   # 红
_COLOR_REVIEW = QColor(255, 240, 200)      # 黄
_COLOR_NEW = QColor(200, 255, 200)         # 绿
_COLOR_ERROR = QColor(220, 220, 220)       # 灰

from .base_worker import BaseWorker

# ── DedupScanWorker ────────────────────────────────────────────

class DedupInitWorker(BaseWorker):
    """后台初始化 Deduplicator（加载 CLIP 模型 + FAISS 索引）。"""

    status = Signal(str)
    finished_ok = Signal(object, list)   # (Deduplicator, folders)

    def __init__(self, download_dir: Path, image_exts: set[str]):
        super().__init__()
        self._download_dir = download_dir
        self._image_exts = image_exts

    def _run(self) -> None:
        self.status.emit("正在初始化查重引擎...")
        dedup = Deduplicator()
        dedup.initialize()

        # 扫描新文件夹（I/O 密集，也在后台完成）
        self.status.emit("正在扫描文件夹...")
        registered = dedup.get_registered_folders()
        folders: list[Path] = []
        if self._download_dir.exists():
            for sub in sorted(self._download_dir.iterdir()):
                if not sub.is_dir():
                    continue
                has_images = any(
                    f.is_file() and f.suffix.lower() in self._image_exts
                    for f in sub.iterdir()
                )
                if has_images and str(sub) not in registered:
                    folders.append(sub)

        self.finished_ok.emit(dedup, folders)


class DedupScanWorker(BaseWorker):
    """后台批量扫描查重。"""

    status = Signal(str)
    folder_started = Signal(int, int, str)   # (idx, total, folder_name)
    image_progress = Signal(int, int)        # (current, total)
    folder_done = Signal(int, int, object)   # (idx, total, DedupScanItem)
    finished_ok = Signal(list)               # list[DedupScanItem]

    def __init__(self, deduplicator: Deduplicator, folders: list[Path]):
        super().__init__()
        self._dedup = deduplicator
        self._folders = folders

    def _run(self) -> None:
        results = self._dedup.batch_scan(
            self._folders,
            is_cancelled=lambda: self._cancelled,
            on_folder_start=lambda i, t, p: self.folder_started.emit(i, t, p.name),
            on_status=lambda msg: self.status.emit(msg),
            on_image_progress=lambda c, t: self.image_progress.emit(c, t),
            on_folder_done=lambda i, t, item: self.folder_done.emit(i, t, item),
        )
        self.finished_ok.emit(results)


# ── DedupRegisterWorker ───────────────────────────────────────

class DedupRegisterWorker(BaseWorker):
    """后台批量注册新产品到 DB + FAISS。"""

    progress = Signal(int, int)     # (current, total)
    finished_ok = Signal(int)       # 注册成功数量

    def __init__(self, deduplicator: Deduplicator, items: list[DedupScanItem]):
        super().__init__()
        self._dedup = deduplicator
        self._items = items

    def _run(self) -> None:
        count = 0
        errors = 0
        total = len(self._items)
        today = date.today().isoformat()
        batch_size = config.REGISTER_BATCH_SIZE

        # 收集有效项
        valid_items = [
            item for item in self._items
            if item.embedding is not None
        ]

        # 分批注册（每批 batch_size 个，单事务 + 单次 FAISS 写入）
        for batch_start in range(0, len(valid_items), batch_size):
            if self._cancelled:
                break

            batch = valid_items[batch_start:batch_start + batch_size]
            batch_tuples = [
                (item.name, "", str(item.folder), "", today,
                 item.image_count, item.embedding)
                for item in batch
            ]

            try:
                self._dedup.register_products_batch(batch_tuples)
                count += len(batch)
            except Exception as e:
                # 批量失败时回退到逐个注册
                logger.warning("批量注册失败，回退逐个注册: %s", e)
                for item in batch:
                    if self._cancelled:
                        break
                    try:
                        self._dedup.register_product(
                            name=item.name, store="",
                            folder=str(item.folder), source_url="",
                            download_date=today,
                            image_count=item.image_count,
                            embedding=item.embedding,
                            save=False,
                        )
                        count += 1
                    except Exception as e2:
                        errors += 1
                        logger.warning("注册产品失败 %s: %s", item.name, e2)

            self.progress.emit(
                min(batch_start + len(batch), total), total
            )

        # 最终持久化 FAISS 索引
        self._dedup.save_index()

        if errors > 0:
            self.finished_err.emit(
                f"注册完成但有 {errors} 个失败（成功 {count} 个）"
            )
        else:
            self.finished_ok.emit(count)


# ── ProductImageDialog ────────────────────────────────────────

class ProductImageDialog(QDialog):
    """查看产品所有图片的弹窗。"""

    def __init__(self, folder: Path, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowMinMaxButtonsHint
        )
        self.setWindowTitle(f"产品图片 — {folder.name}")
        self.setMinimumSize(600, 400)
        self.resize(800, 600)

        layout = QVBoxLayout(self)

        info = QLabel(f"文件夹: {folder}")
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.setWordWrap(True)
        layout.addWidget(info)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self._flow = FlowLayout(content, spacing=6)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        images = list_images(folder)
        count_label = QLabel(f"共 {len(images)} 张图片")
        layout.insertWidget(1, count_label)

        for img_path in images:
            thumb = ThumbnailWidget(img_path, THUMBNAIL_SIZE)
            self._flow.addWidget(thumb)

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)


# ── DedupTableModel ──────────────────────────────────────────

class DedupTableModel(VirtualTableModel):
    """查重结果表格模型。"""

    _HEADERS = ["状态", "文件夹名称", "图片数", "相似度", "匹配产品", "操作"]

    # 额外的状态标记（注册后覆盖原始 status 文本）
    # row → override text, 不修改原始数据
    def __init__(self, parent=None):
        super().__init__(parent)
        self._registered_rows: set[int] = set()   # 已入库的行
        self._deleted_rows: dict[int, str] = {}   # 已删除的行 → 标签

    @property
    def _headers(self) -> list[str]:
        return self._HEADERS

    def _column_data(self, row, col, role):
        item: DedupScanItem = self._items[row]

        # 背景色
        if role == Qt.ItemDataRole.BackgroundRole:
            return self._get_bg_color(row, item)

        # 文本对齐
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col in (0, 2, 3):
                return int(Qt.AlignmentFlag.AlignCenter)
            return None

        # 原始数据
        if role == ROLE_RAW_DATA:
            return item

        # 操作列用自定义角色
        if col == 5:
            return self._action_data(row, item, role)

        # DisplayRole
        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return self._status_text(row, item)
            elif col == 1:
                return item.name
            elif col == 2:
                return str(item.image_count)
            elif col == 3:
                if item.best_match:
                    return f"{item.best_match.similarity:.4f}"
                return "—"
            elif col == 4:
                if item.best_match:
                    return item.best_match.existing_product.name
                return "—"

        # ForegroundRole (状态列)
        if role == Qt.ItemDataRole.ForegroundRole and col == 0:
            return self._status_color(row, item)

        # ToolTipRole
        if role == Qt.ItemDataRole.ToolTipRole:
            if col == 0:
                return self._status_tooltip(row, item)

        return None

    def _status_text(self, row: int, item: DedupScanItem) -> str:
        if row in self._deleted_rows:
            return self._deleted_rows[row]
        if row in self._registered_rows:
            return "已入库"
        if item.error:
            return "X"
        if item.status == DedupStatus.DUPLICATE:
            return "重复"
        if item.status == DedupStatus.REVIEW:
            return "待审"
        return "新"

    def _status_color(self, row: int, item: DedupScanItem) -> QColor | None:
        if row in self._deleted_rows:
            return QColor(Qt.GlobalColor.gray)
        if row in self._registered_rows:
            return QColor(Qt.GlobalColor.darkGreen)
        if item.error:
            return QColor(Qt.GlobalColor.red)
        if item.status == DedupStatus.DUPLICATE:
            return QColor(Qt.GlobalColor.red)
        if item.status == DedupStatus.REVIEW:
            return QColor(Qt.GlobalColor.darkYellow)
        return QColor(Qt.GlobalColor.darkGreen)

    def _status_tooltip(self, row: int, item: DedupScanItem) -> str | None:
        if item.error:
            return item.error
        if item.status == DedupStatus.DUPLICATE:
            return "自动标记重复"
        if item.status == DedupStatus.REVIEW:
            return "需人工审核"
        return "新产品"

    def _get_bg_color(self, row: int, item: DedupScanItem) -> QColor | None:
        if row in self._deleted_rows:
            return _COLOR_ERROR
        if item.error:
            return _COLOR_ERROR
        if item.status == DedupStatus.DUPLICATE:
            return _COLOR_DUPLICATE
        if item.status == DedupStatus.REVIEW:
            return _COLOR_REVIEW
        if item.status == DedupStatus.NEW:
            return _COLOR_NEW
        return None

    def _action_data(self, row: int, item: DedupScanItem, role: int):
        if role == ROLE_ACTION_TYPE:
            if row in self._deleted_rows:
                return "label"
            if row in self._registered_rows:
                return "label"
            if item.error:
                return "label"
            if item.status in (DedupStatus.DUPLICATE, DedupStatus.REVIEW):
                return "button"
            return "label"

        if role == ROLE_ACTION_TEXT:
            if row in self._deleted_rows:
                return self._deleted_rows[row]
            if row in self._registered_rows:
                return "已入库"
            if item.error:
                return "出错"
            if item.status in (DedupStatus.DUPLICATE, DedupStatus.REVIEW):
                return "审核"
            return "新产品"

        if role == ROLE_ACTION_COLOR:
            if row in self._deleted_rows:
                return "gray"
            if row in self._registered_rows:
                return "green"
            if item.error:
                return "red"
            if item.status == DedupStatus.NEW:
                return "green"
            return None

        if role == ROLE_ACTION_TOOLTIP:
            if item.error:
                return item.error
            return None

        return None

    def mark_registered(self, row: int) -> None:
        """标记为已入库。"""
        self._registered_rows.add(row)
        left = self.index(row, 0)
        right = self.index(row, self.columnCount() - 1)
        self.dataChanged.emit(left, right)

    def mark_deleted(self, row: int, label: str) -> None:
        """标记为已删除。"""
        self._deleted_rows[row] = label
        left = self.index(row, 0)
        right = self.index(row, self.columnCount() - 1)
        self.dataChanged.emit(left, right)

    def mark_all_new_registered(self) -> None:
        """批量标记所有 NEW 行为已入库。"""
        for row, item in enumerate(self._items):
            if (row not in self._registered_rows
                    and row not in self._deleted_rows
                    and item.status == DedupStatus.NEW
                    and not item.error
                    and item.embedding is not None):
                self._registered_rows.add(row)
        # 批量通知
        if self._items:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(len(self._items) - 1, self.columnCount() - 1),
            )

    def remove_item(self, row: int) -> None:
        """删除指定行，同时调整索引集合避免状态错乱。"""
        if row < 0 or row >= len(self._items):
            return
        # 调整 _registered_rows：移除该行，大于该行的索引减 1
        new_registered: set[int] = set()
        for r in self._registered_rows:
            if r < row:
                new_registered.add(r)
            elif r > row:
                new_registered.add(r - 1)
            # r == row: 丢弃
        self._registered_rows = new_registered

        # 调整 _deleted_rows：同理
        new_deleted: dict[int, str] = {}
        for r, label in self._deleted_rows.items():
            if r < row:
                new_deleted[r] = label
            elif r > row:
                new_deleted[r - 1] = label
        self._deleted_rows = new_deleted

        super().remove_item(row)

    def clear(self) -> None:
        self._registered_rows.clear()
        self._deleted_rows.clear()
        super().clear()


# ── DedupFilterProxy ─────────────────────────────────────────

class DedupFilterProxy(QSortFilterProxyModel):
    """查重结果筛选代理模型。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filter_id = _FILTER_ALL

    def set_filter(self, filter_id: int) -> None:
        self._filter_id = filter_id
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        if self._filter_id == _FILTER_ALL:
            return True

        model = self.sourceModel()
        item = model.get_item(source_row)
        if item is None:
            return False

        if self._filter_id == _FILTER_DUPLICATE:
            return item.status == DedupStatus.DUPLICATE
        elif self._filter_id == _FILTER_REVIEW:
            return item.status == DedupStatus.REVIEW
        elif self._filter_id == _FILTER_NEW:
            return item.status == DedupStatus.NEW and not item.error
        elif self._filter_id == _FILTER_ERROR:
            return bool(item.error)
        return True


# ── DedupTab ──────────────────────────────────────────────────

class DedupTab(QWidget):
    """查重标签页 — 扫描新文件夹、比对库中已有产品、审核/注册"""

    status_message = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._download_dir: Path = config.DEFAULT_DOWNLOAD_DIR.resolve()
        self._deduplicator: Deduplicator | None = None
        self._init_worker: DedupInitWorker | None = None
        self._scan_worker: DedupScanWorker | None = None
        self._register_worker: DedupRegisterWorker | None = None
        self._folder_scan_worker: BaseWorker | None = None
        self._current_filter = _FILTER_ALL
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

        # ── 控制栏 ────────────────────────────────────────────
        ctrl_layout = QHBoxLayout()

        self._btn_scan = QPushButton("扫描查重")
        self._btn_scan.clicked.connect(self._on_scan)
        ctrl_layout.addWidget(self._btn_scan)

        self._btn_stop = QPushButton("停止")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        ctrl_layout.addWidget(self._btn_stop)

        ctrl_layout.addStretch()

        self._btn_browse_db = QPushButton("浏览特征库")
        self._btn_browse_db.setEnabled(False)
        self._btn_browse_db.clicked.connect(self._on_browse_db)
        ctrl_layout.addWidget(self._btn_browse_db)

        self._db_count_label = QLabel("特征库: 已收录 0 个产品")
        ctrl_layout.addWidget(self._db_count_label)

        main_layout.addLayout(ctrl_layout)

        # ── 进度条 ────────────────────────────────────────────
        self._progress = StatusProgressBar()
        main_layout.addWidget(self._progress)

        # ── 筛选栏 ────────────────────────────────────────────
        filter_layout = QHBoxLayout()

        self._btn_filter_all = QPushButton("全部")
        self._btn_filter_dup = QPushButton("重复")
        self._btn_filter_review = QPushButton("待审核")
        self._btn_filter_new = QPushButton("新产品")
        self._btn_filter_err = QPushButton("出错")

        for btn in (self._btn_filter_all, self._btn_filter_dup,
                    self._btn_filter_review, self._btn_filter_new,
                    self._btn_filter_err):
            btn.setCheckable(True)
            btn.setFixedWidth(72)
        self._btn_filter_all.setChecked(True)

        self._filter_group = QButtonGroup(self)
        self._filter_group.setExclusive(True)
        self._filter_group.addButton(self._btn_filter_all, _FILTER_ALL)
        self._filter_group.addButton(self._btn_filter_dup, _FILTER_DUPLICATE)
        self._filter_group.addButton(self._btn_filter_review, _FILTER_REVIEW)
        self._filter_group.addButton(self._btn_filter_new, _FILTER_NEW)
        self._filter_group.addButton(self._btn_filter_err, _FILTER_ERROR)
        self._filter_group.idClicked.connect(self._on_filter_changed)

        filter_layout.addWidget(self._btn_filter_all)
        filter_layout.addWidget(self._btn_filter_dup)
        filter_layout.addWidget(self._btn_filter_review)
        filter_layout.addWidget(self._btn_filter_new)
        filter_layout.addWidget(self._btn_filter_err)
        filter_layout.addStretch()

        self._summary_label = QLabel("")
        filter_layout.addWidget(self._summary_label)

        main_layout.addLayout(filter_layout)

        # ── 结果表格 (Model/View) ────────────────────────────
        self._model = DedupTableModel(self)
        self._proxy = DedupFilterProxy(self)
        self._proxy.setSourceModel(self._model)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)

        # 操作列委托
        self._action_delegate = ButtonDelegate(
            on_click=self._on_review, parent=self._table
        )
        self._table.setItemDelegateForColumn(5, self._action_delegate)

        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_table_context_menu)
        self._table.doubleClicked.connect(self._on_table_double_click)

        main_layout.addWidget(self._table, 1)

        # ── 底部操作行 ────────────────────────────────────────
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        self._btn_register_all = QPushButton("确认全部新产品入库")
        self._btn_register_all.setEnabled(False)
        self._btn_register_all.clicked.connect(self._on_register_all_new)
        bottom_layout.addWidget(self._btn_register_all)
        main_layout.addLayout(bottom_layout)

    # ── 便捷属性 ─────────────────────────────────────────────

    @property
    def _scan_results(self) -> list[DedupScanItem]:
        """兼容旧代码的访问方式。"""
        return self._model.items

    # ── Deduplicator 懒初始化 ────────────────────────────────

    def _ensure_deduplicator(self) -> Deduplicator | None:
        """返回已初始化的 Deduplicator，若未初始化返回 None。"""
        return self._deduplicator

    def _update_db_count(self) -> None:
        """更新特征库产品数量显示。"""
        if self._deduplicator:
            count = self._deduplicator.product_count
            self._db_count_label.setText(f"特征库: 已收录 {count} 个产品")

    # ── 目录浏览 ───────────────────────────────────────────────

    @Slot()
    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择下载目录", str(self._download_dir),
        )
        if path:
            self._download_dir = Path(path)
            self._dir_label.setText(str(self._download_dir))

    # ── 特征库浏览 ─────────────────────────────────────────────

    @Slot()
    def _on_browse_db(self) -> None:
        if self._deduplicator is None:
            return
        dlg = FeatureStoreBrowserDialog(self._deduplicator, parent=self)
        dlg.exec()
        self._update_db_count()

    # ── 扫描控制 ───────────────────────────────────────────────

    @Slot()
    def _on_scan(self) -> None:
        if not self._download_dir.exists():
            self._progress.set_status("目录不存在")
            self.status_message.emit("目录不存在")
            return

        # 重置状态
        self._model.clear()
        self._update_summary()
        self._btn_register_all.setEnabled(False)
        self._btn_scan.setEnabled(False)
        self._btn_stop.setEnabled(True)

        if self._deduplicator is not None:
            # 已初始化 — 复用现有引擎，直接扫描新文件夹
            self._progress.set_status("正在扫描新文件夹...")
            self._scan_new_folders(self._deduplicator)
        else:
            # 首次 — 后台初始化引擎 + 扫描文件夹
            self._progress.set_status("正在初始化查重引擎（首次可能需要数秒）...")
            self._start_init_worker()

    def _scan_new_folders(self, dedup: Deduplicator) -> None:
        """后台扫描新文件夹，完成后启动扫描 Worker。"""
        self._progress.set_status("正在扫描文件夹...")

        class FolderScanWorker(BaseWorker):
            """后台扫描新文件夹（避免阻塞主线程）。"""
            folders_ready = Signal(list)

            def __init__(self, download_dir: Path, registered: set[str]):
                super().__init__()
                self._download_dir = download_dir
                self._registered = registered

            def _run(self) -> None:
                all_folders = (
                    find_image_folders(self._download_dir)
                    if self._download_dir.exists()
                    else []
                )
                folders = [f for f in all_folders if str(f) not in self._registered]
                self.folders_ready.emit(folders)

        registered = dedup.get_registered_folders()
        self._folder_scan_worker = FolderScanWorker(self._download_dir, registered)

        def on_folders_ready(folders: list) -> None:
            self._folder_scan_worker = None
            if not folders:
                self._progress.set_status("未找到新的产品文件夹")
                self.status_message.emit("未找到新的产品文件夹（所有文件夹已在库中）")
                self._btn_scan.setEnabled(True)
                self._btn_stop.setEnabled(False)
                return
            self._start_scan_worker(dedup, folders)

        self._folder_scan_worker.folders_ready.connect(on_folders_ready)
        self._folder_scan_worker.finished_err.connect(self._on_error)
        self._folder_scan_worker.start()

    def _start_init_worker(self) -> None:
        """启动后台初始化 + 文件夹扫描 Worker。"""
        self._init_worker = DedupInitWorker(self._download_dir, IMAGE_EXTS)
        self._init_worker.status.connect(self._on_worker_status)
        self._init_worker.finished_ok.connect(self._on_init_done)
        self._init_worker.finished_err.connect(self._on_init_error)
        self._init_worker.start()

    @Slot(object, list)
    def _on_init_done(self, dedup: Deduplicator, folders: list) -> None:
        """初始化完成后，如果旧实例存在则关闭，启动扫描。"""
        if self._deduplicator is not None and self._deduplicator is not dedup:
            self._deduplicator.close()
        self._deduplicator = dedup
        self._update_db_count()
        self._btn_browse_db.setEnabled(True)
        self._init_worker = None

        if not folders:
            self._progress.set_status("未找到新的产品文件夹")
            self.status_message.emit("未找到新的产品文件夹（所有文件夹已在库中）")
            self._btn_scan.setEnabled(True)
            self._btn_stop.setEnabled(False)
            return

        self._start_scan_worker(dedup, folders)

    def _start_scan_worker(self, dedup: Deduplicator, folders: list[Path]) -> None:
        """启动扫描 Worker。"""
        self._scan_worker = DedupScanWorker(dedup, folders)
        self._scan_worker.status.connect(self._on_worker_status)
        self._scan_worker.folder_started.connect(self._on_folder_started)
        self._scan_worker.image_progress.connect(self._on_image_progress)
        self._scan_worker.folder_done.connect(self._on_folder_done)
        self._scan_worker.finished_ok.connect(self._on_finished)
        self._scan_worker.finished_err.connect(self._on_error)
        self._scan_worker.start()

    @Slot(str)
    def _on_init_error(self, error: str) -> None:
        self._btn_scan.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.set_status(f"初始化失败: {error}")
        self.status_message.emit(f"查重引擎初始化失败: {error}")
        self._init_worker = None

    @Slot()
    def _on_stop(self) -> None:
        if self._scan_worker:
            self._scan_worker.cancel()
            self._btn_stop.setEnabled(False)
            self._progress.set_status("正在停止...")

    # ── Worker 信号处理 ─────────────────────────────────────────

    @Slot(str)
    def _on_worker_status(self, msg: str) -> None:
        self._progress.set_status(msg)

    @Slot(int, int, str)
    def _on_folder_started(self, idx: int, total: int, name: str) -> None:
        self._progress.set_status(f"正在扫描: {name}")
        self._progress.set_progress(idx, total)

    @Slot(int, int)
    def _on_image_progress(self, current: int, total: int) -> None:
        self._progress.set_stats(f"图片 {current}/{total}")

    @Slot(int, int, object)
    def _on_folder_done(self, idx: int, total: int, item: DedupScanItem) -> None:
        self._model.append_item(item)
        self._progress.set_progress(idx + 1, total)
        self._update_summary()

    @Slot(list)
    def _on_finished(self, results: list) -> None:
        self._btn_scan.setEnabled(True)
        self._btn_stop.setEnabled(False)
        items = self._scan_results
        n = len(items)
        n_dup = sum(1 for r in items if r.status == DedupStatus.DUPLICATE)
        n_review = sum(1 for r in items if r.status == DedupStatus.REVIEW)
        n_new = sum(1 for r in items
                    if r.status == DedupStatus.NEW and not r.error)

        self._progress.set_status(
            f"扫描完成: {n} 个文件夹, {n_dup} 重复, {n_review} 待审核, {n_new} 新产品"
        )
        self._progress.set_stats("")
        self._progress.set_progress(n, n)
        self.status_message.emit(f"查重完成: {n_new} 新 / {n_dup} 重复 / {n_review} 待审核")
        self._scan_worker = None
        self._update_db_count()

        # 有新产品时启用批量入库按钮
        if n_new > 0:
            self._btn_register_all.setEnabled(True)

    @Slot(str)
    def _on_error(self, error: str) -> None:
        self._btn_scan.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.set_status(f"扫描出错: {error}")
        self.status_message.emit(f"查重出错: {error}")
        self._scan_worker = None

    # ── 筛选 ───────────────────────────────────────────────────

    @Slot(int)
    def _on_filter_changed(self, filter_id: int) -> None:
        self._current_filter = filter_id
        self._proxy.set_filter(filter_id)

    def _update_summary(self) -> None:
        items = self._scan_results
        total = len(items)
        n_dup = sum(1 for r in items if r.status == DedupStatus.DUPLICATE)
        n_review = sum(1 for r in items if r.status == DedupStatus.REVIEW)
        n_new = sum(1 for r in items
                    if r.status == DedupStatus.NEW and not r.error)
        n_err = sum(1 for r in items if r.error)
        self._summary_label.setText(
            f"共 {total} | 重复 {n_dup} | 待审 {n_review} | 新 {n_new} | 错 {n_err}"
        )

    # ── 查看产品图片 ─────────────────────────────────────────────

    @Slot()
    def _on_table_double_click(self, index: QModelIndex) -> None:
        source_index = self._proxy.mapToSource(index)
        self._show_product_images(source_index.row())

    @Slot()
    def _on_table_context_menu(self, pos) -> None:
        index = self._table.indexAt(pos)
        if not index.isValid():
            return
        source_row = self._proxy.mapToSource(index).row()
        if source_row < 0 or source_row >= len(self._scan_results):
            return
        menu = QMenu(self)
        act_view = QAction("查看图片", self)
        act_view.triggered.connect(lambda: self._show_product_images(source_row))
        menu.addAction(act_view)

        act_split = QAction("自定义拆分", self)
        act_split.triggered.connect(lambda: self._on_custom_split(source_row))
        menu.addAction(act_split)

        menu.exec(self._table.mapToGlobal(pos))

    def _show_product_images(self, row: int) -> None:
        if row < 0 or row >= len(self._scan_results):
            return
        item = self._scan_results[row]
        if not item.folder.exists():
            QMessageBox.warning(self, "文件夹不存在", f"文件夹已被删除:\n{item.folder}")
            return
        dlg = ProductImageDialog(item.folder, parent=self)
        dlg.exec()

    def _on_custom_split(self, row: int) -> None:
        """自定义拆分功能：打开拆分对话框对相册进行手动拆分"""
        if row < 0 or row >= len(self._scan_results):
            return
        original_item = self._scan_results[row]
        if not original_item.folder.exists():
            QMessageBox.warning(self, "文件夹不存在", f"文件夹已被删除:\n{original_item.folder}")
            return

        from .split_dialog import SplitDialog
        from ..config import CLUSTER_THRESHOLD_DEFAULT

        dlg = SplitDialog(
            original_item.folder,
            threshold=CLUSTER_THRESHOLD_DEFAULT,
            parent=self,
        )
        result = dlg.exec()

        if result == SplitDialog.DialogCode.Accepted:
            # 拆分完成后，从扫描结果中移除该行
            self._model.remove_item(row)
            self._update_summary()
            self.status_message.emit(f"已拆分: {original_item.name}")

    # ── 审核 ───────────────────────────────────────────────────

    def _on_review(self, row: int) -> None:
        if row >= len(self._scan_results):
            return
        item = self._scan_results[row]

        dlg = DedupReviewDialog(item, parent=self)
        result = dlg.exec()

        if result == DELETE_NEW:
            # 删除新采集产品（左侧）
            self._delete_folder_and_mark(row, item.folder, "已删除")
        elif result == DELETE_EXISTING:
            # 删除已有产品（右侧），并注册新产品入库替代
            match = dlg.selected_match
            if match:
                self._replace_existing(row, match)
        elif result == DedupReviewDialog.DialogCode.Accepted:
            # 不是重复 — 注册入库
            self._register_single(row)

    def _delete_folder_and_mark(self, row: int, folder: Path, label: str) -> None:
        """删除文件夹并更新表格行状态。"""
        # 安全校验：确保路径在下载目录内，防止误删
        try:
            folder.resolve().relative_to(self._download_dir.resolve())
        except ValueError:
            QMessageBox.critical(
                self, "安全限制",
                f"拒绝删除下载目录以外的文件夹:\n{folder}",
            )
            return
        ret = QMessageBox.question(
            self, "确认删除",
            f"确定要删除文件夹及其所有图片吗？\n\n{folder}\n\n此操作不可撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        try:
            if folder.exists():
                shutil.rmtree(folder)
        except OSError as e:
            self._progress.set_status(f"删除失败: {e}")
            return

        self._model.mark_deleted(row, label)
        self._update_summary()
        self.status_message.emit(f"{label}: {self._scan_results[row].name}")

    def _replace_existing(self, row: int, match: DedupMatch) -> None:
        """删除已有产品，注册新产品替代。"""
        item = self._scan_results[row]
        existing = match.existing_product
        dedup = self._deduplicator
        if dedup is None:
            return

        exist_folder = Path(existing.folder)
        ret = QMessageBox.question(
            self, "确认替换",
            f"将删除已有产品并用新产品替代：\n\n"
            f"删除: {existing.name}\n{exist_folder}\n\n"
            f"替换为: {item.name}\n\n此操作不可撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return

        # 1. 从 DB + FAISS 移除已有产品
        dedup.remove_product(existing.id)

        # 2. 删除已有产品的文件夹（校验路径安全性）
        try:
            exist_folder.resolve().relative_to(self._download_dir.resolve())
        except ValueError:
            self._progress.set_status(f"安全限制: 拒绝删除下载目录外的文件夹")
            logger.warning("拒绝删除下载目录外的文件夹: %s", exist_folder)
        else:
            try:
                if exist_folder.exists():
                    shutil.rmtree(exist_folder)
            except OSError as e:
                self._progress.set_status(f"删除旧文件夹失败: {e}")

        # 3. 注册新产品入库
        self._register_single(row)
        self.status_message.emit(f"已替换: {existing.name} → {item.name}")

    def _register_single(self, row: int) -> None:
        """注册单个产品到库中。"""
        item = self._scan_results[row]
        if item.embedding is None:
            return

        dedup = self._deduplicator
        if dedup is None:
            return
        today = date.today().isoformat()

        dedup.register_product(
            name=item.name,
            store="",
            folder=str(item.folder),
            source_url="",
            download_date=today,
            image_count=item.image_count,
            embedding=item.embedding,
        )

        self._model.mark_registered(row)
        self._update_db_count()
        self._update_summary()
        self.status_message.emit(f"已入库: {item.name}")

    # ── 批量注册 ───────────────────────────────────────────────

    @Slot()
    def _on_register_all_new(self) -> None:
        """启动后台线程批量注册所有 NEW 状态的产品。"""
        registered = self._model._registered_rows
        deleted = self._model._deleted_rows
        new_items = [
            item for row, item in enumerate(self._scan_results)
            if item.status == DedupStatus.NEW
            and not item.error
            and item.embedding is not None
            and row not in registered
            and row not in deleted
        ]
        if not new_items:
            self._progress.set_status("没有需要入库的新产品")
            return

        dedup = self._deduplicator
        if dedup is None:
            return
        self._btn_register_all.setEnabled(False)
        self._btn_scan.setEnabled(False)
        self._table.setEnabled(False)
        self._progress.set_status(f"正在注册 {len(new_items)} 个新产品...")

        self._register_worker = DedupRegisterWorker(dedup, new_items)
        self._register_worker.progress.connect(self._on_register_progress)
        self._register_worker.finished_ok.connect(self._on_register_done)
        self._register_worker.finished_err.connect(self._on_register_error)
        self._register_worker.start()

    @Slot(int, int)
    def _on_register_progress(self, current: int, total: int) -> None:
        self._progress.set_progress(current, total)

    @Slot(int)
    def _on_register_done(self, count: int) -> None:
        self._btn_scan.setEnabled(True)
        self._table.setEnabled(True)
        self._progress.set_status(f"注册完成: {count} 个新产品已入库")
        self._progress.set_progress(count, count)
        self.status_message.emit(f"批量入库完成: {count} 个产品")
        self._register_worker = None

        self._model.mark_all_new_registered()
        self._update_db_count()

    @Slot(str)
    def _on_register_error(self, error: str) -> None:
        self._btn_scan.setEnabled(True)
        self._btn_register_all.setEnabled(True)
        self._table.setEnabled(True)
        self._progress.set_status(f"注册出错: {error}")
        self.status_message.emit(f"批量入库出错: {error}")
        self._register_worker = None

    # ── 清理 ───────────────────────────────────────────────────

    def cleanup(self) -> None:
        """应用关闭时调用，持久化数据。"""
        if self._init_worker and self._init_worker.isRunning():
            self._init_worker.wait(3000)
        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.cancel()
            self._scan_worker.wait(3000)
        if self._register_worker and self._register_worker.isRunning():
            self._register_worker.cancel()
            self._register_worker.wait(3000)
        if self._folder_scan_worker and self._folder_scan_worker.isRunning():
            self._folder_scan_worker.cancel()
            self._folder_scan_worker.wait(3000)
        if self._deduplicator:
            self._deduplicator.close()
            self._deduplicator = None
