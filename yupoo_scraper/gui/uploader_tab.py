"""上架标签页 — 微店自动上架 GUI"""

from __future__ import annotations

import asyncio
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import config
from ..chrome_launcher import is_cdp_available, launch_chrome
from ..title_generator import ProductInfo, TitleGenerator
from ..uploader import UploadResult, WeidianUploader
from .widgets import StatusProgressBar

from ..config import IMAGE_EXTS
from ..organizer import find_image_folders
from .base_worker import BaseWorker


# ── Workers ──────────────────────────────────────────────────────

class PreviewWorker(BaseWorker):
    """后台生成标题预览（CLIP zero-shot 分类）。"""

    status = Signal(str)
    progress = Signal(int, int)           # (current, total)
    finished_ok = Signal(list)            # list[ProductInfo]

    def __init__(
        self,
        folders: list[Path],
        price: float,
        stock: int,
        price_mode: str,        # "fixed" | "random"
        price_min: float,
        price_max: float,
    ):
        super().__init__()
        self._folders = folders
        self._price = price
        self._stock = stock
        self._price_mode = price_mode
        self._price_min = price_min
        self._price_max = price_max

    def _run(self) -> None:
        self.status.emit("正在加载 CLIP 模型...")
        gen = TitleGenerator()
        products = gen.batch_generate(
            self._folders,
            price=self._price,
            stock=self._stock,
            on_progress=lambda c, t: self.progress.emit(c, t),
        )

        # 随机价格模式
        if self._price_mode == "random":
            for p in products:
                p.price = round(
                    random.uniform(self._price_min, self._price_max), 2,
                )

        self.finished_ok.emit(products)


class UploadWorker(BaseWorker):
    """后台执行批量上架（Playwright 自动化）。"""

    status = Signal(str)
    progress = Signal(int, int)           # (current, total)
    step_update = Signal(str)             # 当前步骤描述
    product_done = Signal(int, object)    # (index, UploadResult)
    finished_ok = Signal(list)            # list[UploadResult]

    def __init__(self, products: list[ProductInfo], cdp_url: str):
        super().__init__()
        self._products = products
        self._cdp_url = cdp_url
        self._chrome_proc = None

    def cleanup_chrome(self) -> None:
        """清理由本 worker 启动的 Chrome 进程（线程安全）。"""
        proc = self._chrome_proc
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except OSError:
                pass
            self._chrome_proc = None

    def run(self) -> None:
        # UploadWorker 需要自己的事件循环（Playwright 是 async 的），
        # 因此覆盖 run() 而非使用基类的模板方法。
        if sys.platform == "win32":
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(self._run_async())
            self.finished_ok.emit(results)
        except Exception as e:
            self.finished_err.emit(str(e))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    async def _run_async(self) -> list[UploadResult]:
        uploader = WeidianUploader()

        # 自动启动 Chrome（如果未运行）
        self.step_update.emit("正在检查浏览器...")
        if not is_cdp_available():
            self.step_update.emit("正在启动 Chrome 浏览器...")
            try:
                self._chrome_proc = launch_chrome()
            except (FileNotFoundError, TimeoutError) as e:
                raise RuntimeError(str(e)) from e

        self.step_update.emit("正在连接浏览器...")
        try:
            await uploader.connect(self._cdp_url)
        except Exception as e:
            raise RuntimeError(f"浏览器连接失败: {e}") from e

        try:
            results: list[UploadResult] = []
            total = len(self._products)

            for i, product in enumerate(self._products):
                if self._cancelled:
                    for p in self._products[i:]:
                        r = UploadResult(p.folder, False, "已取消", title=p.title)
                        results.append(r)
                        self.product_done.emit(len(results) - 1, r)
                    break

                self.step_update.emit(
                    f"({i + 1}/{total}) {product.title[:20]}..."
                )
                result = await uploader.upload_product(
                    product,
                    on_step=lambda msg: self.step_update.emit(msg),
                )
                results.append(result)
                self.product_done.emit(i, result)
                self.progress.emit(i + 1, total)

            return results
        finally:
            await uploader.disconnect()


# ── UploaderTab ──────────────────────────────────────────────────

class UploaderTab(QWidget):
    """微店上架标签页。"""

    status_message = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._download_dir: Path = config.DEFAULT_DOWNLOAD_DIR.resolve()
        self._products: list[ProductInfo] = []
        self._preview_worker: PreviewWorker | None = None
        self._upload_worker: UploadWorker | None = None
        self._upload_marks: dict = {}
        self._load_upload_marks()
        self._build_ui()

    # ── UI 构建 ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # ── 上架设置区 ─────────────────────────────────────────────
        # 产品目录
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("产品目录:"))
        self._dir_label = QLabel(str(self._download_dir))
        self._dir_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        btn_browse = QPushButton("浏览...")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._on_browse)
        self._btn_refresh = QPushButton("刷新")
        self._btn_refresh.setFixedWidth(60)
        self._btn_refresh.clicked.connect(self._on_refresh)
        dir_layout.addWidget(self._dir_label, 1)
        dir_layout.addWidget(btn_browse)
        dir_layout.addWidget(self._btn_refresh)
        main_layout.addLayout(dir_layout)

        # 价格模式
        price_layout = QHBoxLayout()
        price_layout.addWidget(QLabel("价格模式:"))

        self._price_group = QButtonGroup(self)
        self._rb_fixed = QRadioButton("统一价格")
        self._rb_fixed.setChecked(True)
        self._rb_random = QRadioButton("随机区间")
        self._price_group.addButton(self._rb_fixed)
        self._price_group.addButton(self._rb_random)

        price_layout.addWidget(self._rb_fixed)
        self._fixed_price_input = QLineEdit("299")
        self._fixed_price_input.setFixedWidth(80)
        self._fixed_price_input.setPlaceholderText("¥")
        price_layout.addWidget(self._fixed_price_input)

        price_layout.addSpacing(16)
        price_layout.addWidget(self._rb_random)
        self._price_min_input = QLineEdit("199")
        self._price_min_input.setFixedWidth(60)
        self._price_min_input.setPlaceholderText("¥最低")
        price_layout.addWidget(self._price_min_input)
        price_layout.addWidget(QLabel("~"))
        self._price_max_input = QLineEdit("599")
        self._price_max_input.setFixedWidth(60)
        self._price_max_input.setPlaceholderText("¥最高")
        price_layout.addWidget(self._price_max_input)

        price_layout.addSpacing(16)
        price_layout.addWidget(QLabel("库存:"))
        self._stock_spin = QSpinBox()
        self._stock_spin.setRange(1, 99999)
        self._stock_spin.setValue(999)
        self._stock_spin.setFixedWidth(80)
        price_layout.addWidget(self._stock_spin)

        price_layout.addStretch()

        self._btn_preview = QPushButton("生成预览")
        self._btn_preview.clicked.connect(self._on_preview)
        price_layout.addWidget(self._btn_preview)

        main_layout.addLayout(price_layout)

        # 切换价格输入状态
        self._rb_fixed.toggled.connect(self._on_price_mode_changed)
        self._on_price_mode_changed()

        # ── 产品列表表格 ──────────────────────────────────────────
        main_layout.addWidget(QLabel("产品列表（双击可编辑标题/价格）:"))

        self._product_table = QTableWidget(0, 7)
        self._product_table.setHorizontalHeaderLabels(
            ["#", "☑", "标题", "价格", "主图", "详情", "状态"]
        )
        self._product_table.verticalHeader().setVisible(False)
        self._product_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )

        header = self._product_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)

        main_layout.addWidget(self._product_table, 1)

        # 全选/取消全选
        sel_layout = QHBoxLayout()
        btn_sel_all = QPushButton("全选")
        btn_sel_all.setFixedWidth(60)
        btn_sel_all.clicked.connect(self._on_select_all)
        btn_desel = QPushButton("取消全选")
        btn_desel.setFixedWidth(80)
        btn_desel.clicked.connect(self._on_deselect_all)
        btn_sel_not_uploaded = QPushButton("全选未上架")
        btn_sel_not_uploaded.setFixedWidth(90)
        btn_sel_not_uploaded.clicked.connect(self._on_select_not_uploaded)
        btn_sel_uploaded = QPushButton("全选已上架")
        btn_sel_uploaded.setFixedWidth(90)
        btn_sel_uploaded.clicked.connect(self._on_select_uploaded)
        btn_clear_marks = QPushButton("清除上架标记")
        btn_clear_marks.setFixedWidth(110)
        btn_clear_marks.clicked.connect(self._on_clear_marks)
        sel_layout.addWidget(btn_sel_all)
        sel_layout.addWidget(btn_desel)
        sel_layout.addWidget(btn_sel_not_uploaded)
        sel_layout.addWidget(btn_sel_uploaded)
        btn_delete_sel = QPushButton("删除选中")
        btn_delete_sel.setFixedWidth(80)
        btn_delete_sel.clicked.connect(self._on_delete_selected)
        sel_layout.addWidget(btn_clear_marks)
        sel_layout.addWidget(btn_delete_sel)
        sel_layout.addStretch()
        main_layout.addLayout(sel_layout)

        # ── 浏览器连接 + 控制 ─────────────────────────────────────
        conn_layout = QHBoxLayout()
        conn_layout.addWidget(QLabel("CDP 地址:"))
        self._cdp_input = QLineEdit(config.WEIDIAN_CDP_URL)
        self._cdp_input.setFixedWidth(200)
        conn_layout.addWidget(self._cdp_input)

        conn_layout.addSpacing(16)

        self._btn_upload = QPushButton("开始上架")
        self._btn_upload.clicked.connect(self._on_upload)
        conn_layout.addWidget(self._btn_upload)

        self._btn_stop = QPushButton("停止")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        conn_layout.addWidget(self._btn_stop)

        conn_layout.addStretch()
        main_layout.addLayout(conn_layout)

        # ── 进度条 ────────────────────────────────────────────────
        self._progress = StatusProgressBar()
        main_layout.addWidget(self._progress)

        # ── 结果日志表格 ──────────────────────────────────────────
        main_layout.addWidget(QLabel("上架结果:"))
        self._result_table = QTableWidget(0, 3)
        self._result_table.setHorizontalHeaderLabels(["状态", "产品名称", "备注"])
        self._result_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._result_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._result_table.verticalHeader().setVisible(False)

        rh = self._result_table.horizontalHeader()
        rh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        rh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        rh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        main_layout.addWidget(self._result_table)

    # ── 价格模式切换 ────────────────────────────────────────────────

    @Slot()
    def _on_price_mode_changed(self) -> None:
        fixed = self._rb_fixed.isChecked()
        self._fixed_price_input.setEnabled(fixed)
        self._price_min_input.setEnabled(not fixed)
        self._price_max_input.setEnabled(not fixed)

    # ── 目录浏览 ────────────────────────────────────────────────────

    @Slot()
    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择产品目录", str(self._download_dir),
        )
        if path:
            self._download_dir = Path(path)
            self._dir_label.setText(str(self._download_dir))
            self._on_refresh()

    @Slot()
    def _on_refresh(self) -> None:
        """快速扫描文件夹列表（不运行 CLIP），直接用文件夹名作为临时标题。"""
        if not self._download_dir.exists():
            self._progress.set_status("目录不存在")
            return

        folders = self._find_product_folders(self._download_dir)
        if not folders:
            self._progress.set_status("未找到含图片的子文件夹")
            self._product_table.setRowCount(0)
            self._products.clear()
            return

        # 读取当前价格/库存设置
        try:
            price = float(self._fixed_price_input.text() or "0")
        except ValueError:
            price = 0.0
        stock = self._stock_spin.value()

        self._products.clear()
        for folder in folders:
            images = sorted(
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
            if not images:
                continue

            p_price = price
            if self._rb_random.isChecked():
                try:
                    lo = float(self._price_min_input.text() or "0")
                    hi = float(self._price_max_input.text() or "0")
                except ValueError:
                    lo, hi = 0.0, 0.0
                p_price = round(random.uniform(lo, hi), 2)

            self._products.append(ProductInfo(
                folder=folder,
                title=folder.name,  # 临时标题 = 文件夹名
                price=p_price,
                stock=stock,
                main_images=images[:config.MAIN_IMAGE_MAX],
                detail_images=images[config.MAIN_IMAGE_MAX:],
            ))

        self._populate_product_table()
        msg = f"已加载 {len(self._products)} 个产品文件夹，正在生成标题..."
        self._progress.set_status(msg)
        self.status_message.emit(msg)

        # 自动启动 CLIP 生成标题
        self._on_preview()

    # ── 全选/取消全选 ───────────────────────────────────────────────

    @Slot()
    def _on_select_all(self) -> None:
        for row in range(self._product_table.rowCount()):
            item = self._product_table.item(row, 1)
            if item:
                item.setCheckState(Qt.CheckState.Checked)

    @Slot()
    def _on_deselect_all(self) -> None:
        for row in range(self._product_table.rowCount()):
            item = self._product_table.item(row, 1)
            if item:
                item.setCheckState(Qt.CheckState.Unchecked)

    @Slot()
    def _on_select_not_uploaded(self) -> None:
        """勾选所有未上架的产品，取消勾选已上架的。"""
        for row in range(self._product_table.rowCount()):
            chk = self._product_table.item(row, 1)
            if chk and row < len(self._products):
                uploaded = self._is_uploaded(self._products[row].folder)
                chk.setCheckState(
                    Qt.CheckState.Unchecked if uploaded else Qt.CheckState.Checked
                )

    @Slot()
    def _on_select_uploaded(self) -> None:
        """勾选所有已上架的产品，取消勾选未上架的。"""
        for row in range(self._product_table.rowCount()):
            chk = self._product_table.item(row, 1)
            if chk and row < len(self._products):
                uploaded = self._is_uploaded(self._products[row].folder)
                chk.setCheckState(
                    Qt.CheckState.Checked if uploaded else Qt.CheckState.Unchecked
                )

    # ── 生成预览 ────────────────────────────────────────────────────

    @Slot()
    def _on_preview(self) -> None:
        if not self._download_dir.exists():
            self._progress.set_status("目录不存在")
            self.status_message.emit("目录不存在")
            return

        folders = self._find_product_folders(self._download_dir)
        if not folders:
            self._progress.set_status("未找到含图片的子文件夹")
            self.status_message.emit("未找到含图片的文件夹")
            return

        # 获取价格参数
        price_mode = "fixed" if self._rb_fixed.isChecked() else "random"
        try:
            fixed_price = float(self._fixed_price_input.text() or "0")
        except ValueError:
            fixed_price = 0.0
        try:
            price_min = float(self._price_min_input.text() or "0")
        except ValueError:
            price_min = 0.0
        try:
            price_max = float(self._price_max_input.text() or "0")
        except ValueError:
            price_max = 0.0

        stock = self._stock_spin.value()

        # 取消之前正在运行的预览 worker（防止并发覆盖）
        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.wait(3000)

        self._btn_preview.setEnabled(False)
        self._progress.set_status("正在生成预览...")

        self._preview_worker = PreviewWorker(
            folders,
            price=fixed_price,
            stock=stock,
            price_mode=price_mode,
            price_min=price_min,
            price_max=price_max,
        )
        self._preview_worker.status.connect(self._on_preview_status)
        self._preview_worker.progress.connect(
            lambda c, t: self._progress.set_progress(c, t)
        )
        self._preview_worker.finished_ok.connect(self._on_preview_done)
        self._preview_worker.finished_err.connect(self._on_preview_error)
        self._preview_worker.start()

    @Slot(str)
    def _on_preview_status(self, msg: str) -> None:
        self._progress.set_status(msg)

    @Slot(list)
    def _on_preview_done(self, products: list) -> None:
        self._products = products
        self._populate_product_table()
        self._btn_preview.setEnabled(True)
        msg = f"预览完成: {len(products)} 个产品"
        self._progress.set_status(msg)
        self._progress.set_progress(len(products), len(products))
        self.status_message.emit(msg)
        self._preview_worker = None

    @Slot(str)
    def _on_preview_error(self, error: str) -> None:
        self._btn_preview.setEnabled(True)
        self._progress.set_status(f"预览出错: {error}")
        self.status_message.emit(f"预览出错: {error}")
        self._preview_worker = None

    def _populate_product_table(self) -> None:
        """用 ProductInfo 列表填充产品表格。"""
        self._product_table.setRowCount(0)
        self._product_table.setRowCount(len(self._products))

        for row, p in enumerate(self._products):
            # 序号
            idx_item = QTableWidgetItem(str(row + 1))
            idx_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            idx_item.setFlags(idx_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._product_table.setItem(row, 0, idx_item)

            # 勾选框
            chk = QTableWidgetItem()
            chk.setCheckState(Qt.CheckState.Checked)
            chk.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
            )
            self._product_table.setItem(row, 1, chk)

            # 标题（可编辑）
            title_item = QTableWidgetItem(p.title)
            self._product_table.setItem(row, 2, title_item)

            # 价格（可编辑）
            price_item = QTableWidgetItem(f"{p.price:.2f}")
            price_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._product_table.setItem(row, 3, price_item)

            # 主图数
            main_item = QTableWidgetItem(str(len(p.main_images)))
            main_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            main_item.setFlags(
                main_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            self._product_table.setItem(row, 4, main_item)

            # 详情图数
            detail_item = QTableWidgetItem(str(len(p.detail_images)))
            detail_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            detail_item.setFlags(
                detail_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            self._product_table.setItem(row, 5, detail_item)

            # 状态列
            status_item = QTableWidgetItem()
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            status_item.setFlags(
                status_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            if self._is_uploaded(p.folder):
                status_item.setText("已上架")
                status_item.setForeground(Qt.GlobalColor.darkGreen)
                chk.setCheckState(Qt.CheckState.Unchecked)
            self._product_table.setItem(row, 6, status_item)

    # ── 上架控制 ────────────────────────────────────────────────────

    @Slot()
    def _on_upload(self) -> None:
        selected = self._get_selected_products()
        if not selected:
            self._progress.set_status("请先选择要上架的产品")
            self.status_message.emit("未选择产品")
            return

        cdp_url = self._cdp_input.text().strip()
        if not cdp_url:
            self._progress.set_status("请输入 CDP 地址")
            return

        # 同步表格编辑回产品数据
        warnings = self._sync_table_to_products()
        if warnings:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "价格数据异常",
                "以下行存在问题：\n" + "\n".join(warnings),
            )

        self._btn_upload.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_preview.setEnabled(False)
        self._result_table.setRowCount(0)
        self._progress.reset()

        self._upload_worker = UploadWorker(selected, cdp_url)
        self._upload_worker.status.connect(self._on_upload_status)
        self._upload_worker.progress.connect(self._on_upload_progress)
        self._upload_worker.step_update.connect(self._on_upload_status)
        self._upload_worker.product_done.connect(self._on_product_done)
        self._upload_worker.finished_ok.connect(self._on_upload_done)
        self._upload_worker.finished_err.connect(self._on_upload_error)
        self._upload_worker.start()

    @Slot(str)
    def _on_upload_status(self, msg: str) -> None:
        self._progress.set_status(msg)

    @Slot(int, int)
    def _on_upload_progress(self, current: int, total: int) -> None:
        self._progress.set_progress(current, total)

    @Slot()
    def _on_stop(self) -> None:
        if self._upload_worker:
            self._upload_worker.cancel()
            self._btn_stop.setEnabled(False)
            self._progress.set_status("正在停止...")

    @Slot(int, object)
    def _on_product_done(self, idx: int, result: UploadResult) -> None:
        """单个产品上架完成，更新结果日志。"""
        row = self._result_table.rowCount()
        self._result_table.insertRow(row)

        # 状态
        status_item = QTableWidgetItem()
        status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if result.success:
            status_item.setText("✓")
            status_item.setForeground(Qt.GlobalColor.darkGreen)
        else:
            status_item.setText("✗")
            status_item.setForeground(Qt.GlobalColor.red)
        self._result_table.setItem(row, 0, status_item)

        # 产品名称（优先显示标题，无标题时回退到文件夹名）
        display_name = result.title if result.title else result.folder.name
        name_item = QTableWidgetItem(display_name)
        self._result_table.setItem(row, 1, name_item)

        # 备注
        note = result.error or "上架成功"
        note_item = QTableWidgetItem(note)
        if result.error:
            note_item.setForeground(Qt.GlobalColor.red)
        self._result_table.setItem(row, 2, note_item)

        # 上架成功 → 标记并更新产品表格
        if result.success:
            self._mark_uploaded(result.folder)
            self._save_upload_marks()
            self._update_product_table_status(result.folder)

    def _cleanup_chrome_proc(self) -> None:
        """清理 UploadWorker 启动的 Chrome 进程。"""
        if self._upload_worker:
            self._upload_worker.cleanup_chrome()

    @Slot(list)
    def _on_upload_done(self, results: list) -> None:
        self._cleanup_chrome_proc()
        self._btn_upload.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_preview.setEnabled(True)
        ok = sum(1 for r in results if r.success)
        fail = len(results) - ok
        msg = f"上架完成: {ok} 成功, {fail} 失败"
        self._progress.set_status(msg)
        self._progress.set_progress(len(results), len(results))
        self.status_message.emit(msg)
        self._upload_worker = None

    @Slot(str)
    def _on_upload_error(self, error: str) -> None:
        self._cleanup_chrome_proc()
        self._btn_upload.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_preview.setEnabled(True)
        self._progress.set_status(f"上架出错: {error}")
        self.status_message.emit(f"上架出错: {error}")
        self._upload_worker = None

    # ── 辅助方法 ────────────────────────────────────────────────────

    def _sync_table_to_products(self) -> list[str]:
        """将表格中用户编辑的标题/价格同步回 ProductInfo。

        Returns:
            包含无效价格警告信息的列表（空列表 = 全部有效）。
        """
        warnings: list[str] = []
        for row in range(self._product_table.rowCount()):
            if row >= len(self._products):
                break
            title_item = self._product_table.item(row, 2)
            if title_item:
                self._products[row].title = title_item.text()
            price_item = self._product_table.item(row, 3)
            if price_item:
                try:
                    price = float(price_item.text())
                    if price <= 0:
                        warnings.append(f"第 {row + 1} 行价格 <= 0")
                    self._products[row].price = price
                except ValueError:
                    warnings.append(
                        f"第 {row + 1} 行价格无效: {price_item.text()!r}"
                    )
        return warnings

    def _get_selected_products(self) -> list[ProductInfo]:
        """获取勾选的产品列表。"""
        selected = []
        for row in range(self._product_table.rowCount()):
            chk = self._product_table.item(row, 1)
            if chk and chk.checkState() == Qt.CheckState.Checked:
                if row < len(self._products):
                    selected.append(self._products[row])
        return selected

    # ── 上架标记 ────────────────────────────────────────────────────

    def _load_upload_marks(self) -> None:
        """从 JSON 文件加载上架标记。"""
        path = config.UPLOAD_MARKS_FILE
        if path.exists():
            try:
                self._upload_marks = json.loads(path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                self._upload_marks = {}
        else:
            self._upload_marks = {}

    def _save_upload_marks(self) -> None:
        """将上架标记写入 JSON 文件。"""
        path = config.UPLOAD_MARKS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._upload_marks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _is_uploaded(self, folder: Path) -> bool:
        dir_key = str(folder.parent)
        return folder.name in self._upload_marks.get(dir_key, {})

    def _mark_uploaded(self, folder: Path) -> None:
        dir_key = str(folder.parent)
        if dir_key not in self._upload_marks:
            self._upload_marks[dir_key] = {}
        self._upload_marks[dir_key][folder.name] = datetime.now().isoformat(
            timespec="seconds"
        )

    def _unmark_uploaded(self, folder: Path) -> None:
        dir_key = str(folder.parent)
        group = self._upload_marks.get(dir_key)
        if group and folder.name in group:
            del group[folder.name]
            if not group:
                del self._upload_marks[dir_key]

    def _update_product_table_status(self, folder: Path) -> None:
        """更新产品表格中对应行的状态列和勾选状态。"""
        for row in range(self._product_table.rowCount()):
            if row < len(self._products) and self._products[row].folder == folder:
                status_item = self._product_table.item(row, 6)
                if status_item is None:
                    status_item = QTableWidgetItem()
                    status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    status_item.setFlags(
                        status_item.flags() & ~Qt.ItemFlag.ItemIsEditable
                    )
                    self._product_table.setItem(row, 6, status_item)
                if self._is_uploaded(folder):
                    status_item.setText("已上架")
                    status_item.setForeground(Qt.GlobalColor.darkGreen)
                    chk = self._product_table.item(row, 1)
                    if chk:
                        chk.setCheckState(Qt.CheckState.Unchecked)
                else:
                    status_item.setText("")
                break

    @Slot()
    def _on_clear_marks(self) -> None:
        """清除上架标记：有勾选 → 只清勾选的；无勾选 → 清当前目录全部。"""
        selected_rows = [
            row for row in range(self._product_table.rowCount())
            if (chk := self._product_table.item(row, 1))
            and chk.checkState() == Qt.CheckState.Checked
        ]

        if selected_rows:
            for row in selected_rows:
                if row < len(self._products):
                    self._unmark_uploaded(self._products[row].folder)
        else:
            dir_key = str(self._download_dir)
            if dir_key in self._upload_marks:
                del self._upload_marks[dir_key]

        self._save_upload_marks()
        self._populate_product_table()
        self._progress.set_status("已清除上架标记")

    @Slot()
    def _on_delete_selected(self) -> None:
        """删除勾选的产品文件夹（含磁盘文件），同时清除对应上架标记。"""
        from PySide6.QtWidgets import QMessageBox

        selected_rows = [
            row for row in range(self._product_table.rowCount())
            if row < len(self._products)
            and (chk := self._product_table.item(row, 1))
            and chk.checkState() == Qt.CheckState.Checked
        ]
        if not selected_rows:
            self._progress.set_status("请先勾选要删除的产品")
            return

        reply = QMessageBox.warning(
            self, "确认删除",
            f"即将永久删除 {len(selected_rows)} 个产品文件夹，此操作不可撤销！\n\n继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        deleted = 0
        errors: list[str] = []
        # 倒序删除，避免索引偏移
        for row in reversed(selected_rows):
            folder = self._products[row].folder
            try:
                shutil.rmtree(folder)
                self._unmark_uploaded(folder)
                self._products.pop(row)
                deleted += 1
            except OSError as e:
                errors.append(f"{folder.name}: {e}")

        self._save_upload_marks()
        self._populate_product_table()

        msg = f"已删除 {deleted} 个产品"
        if errors:
            msg += f"，{len(errors)} 个失败"
            self._progress.set_status(msg)
            QMessageBox.warning(self, "部分删除失败", "\n".join(errors))
        else:
            self._progress.set_status(msg)

    def cleanup(self) -> None:
        """安全停止后台线程。"""
        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.wait(3000)
        if self._upload_worker and self._upload_worker.isRunning():
            self._upload_worker.cancel()
            self._upload_worker.wait(5000)

    @staticmethod
    def _find_product_folders(base_dir: Path) -> list[Path]:
        return find_image_folders(base_dir)
