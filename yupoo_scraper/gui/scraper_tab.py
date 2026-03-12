"""采集页面 — URL 输入、相册列表、下载控制、进度显示"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import aiohttp
import imagehash
from PIL import Image
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

from .. import config
from ..downloader import BatchResult, Downloader
from ..organizer import sanitize_folder_name, unique_folder_path
from ..scraper import (
    Album, AlbumDetail, Category,
    get_album_images, get_album_list, get_categories, get_category_albums,
    parse_username,
)
from .widgets import StatusProgressBar


def _create_connector() -> aiohttp.TCPConnector:
    """创建带连接数限制的 TCP 连接器。"""
    return aiohttp.TCPConnector(
        limit=config.HTTP_CONN_LIMIT,
        limit_per_host=config.HTTP_CONN_LIMIT_PER_HOST,
    )


class ScraperTab(QWidget):
    """采集标签页"""

    # 信号
    status_message = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._albums: list[Album] = []
        self._album_details: list[AlbumDetail] = []
        self._categories: list[Category] = []
        self._username: str = ""
        self._downloader: Downloader | None = None
        self._download_task: asyncio.Task | None = None
        self._fetch_task: asyncio.Task | None = None
        self._download_log: dict = self._load_download_log()
        self._init_ui()

    # ── UI 构建 ────────────────────────────────────────────────

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # -- URL 输入区域 --
        url_group = QGroupBox("Yupoo 地址")
        url_layout = QVBoxLayout(url_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("相册主页:"))
        self._url_input = QComboBox()
        self._url_input.setEditable(True)
        self._url_input.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._url_input.lineEdit().setPlaceholderText(
            "https://x.yupoo.com/photos/username/albums"
        )
        self._load_url_history()
        row1.addWidget(self._url_input, 1)
        self._fetch_btn = QPushButton("获取相册列表")
        self._fetch_btn.clicked.connect(self._on_fetch_albums)
        row1.addWidget(self._fetch_btn)
        url_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("保存路径:"))
        self._dir_input = QLineEdit(str(Path("./downloads").resolve()))
        row2.addWidget(self._dir_input, 1)
        browse_btn = QPushButton("选择")
        browse_btn.clicked.connect(self._on_browse_dir)
        row2.addWidget(browse_btn)
        url_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("分类:"))
        self._category_combo = QComboBox()
        self._category_combo.setEnabled(False)
        self._category_combo.currentIndexChanged.connect(self._on_category_changed)
        row3.addWidget(self._category_combo, 1)
        url_layout.addLayout(row3)

        layout.addWidget(url_group)

        # -- 相册列表 --
        album_group = QGroupBox("相册列表")
        album_layout = QVBoxLayout(album_group)

        self._album_table = QTableWidget(0, 4)
        self._album_table.setHorizontalHeaderLabels(["选择", "相册名称", "图片数", "状态"])
        header = self._album_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self._album_table.setColumnWidth(0, 50)
        self._album_table.setColumnWidth(2, 80)
        self._album_table.setColumnWidth(3, 120)
        self._album_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        album_layout.addWidget(self._album_table)

        # 全选/取消 + 统计
        select_row = QHBoxLayout()
        self._select_all_btn = QPushButton("全选")
        self._select_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        self._deselect_all_btn = QPushButton("取消全选")
        self._deselect_all_btn.clicked.connect(lambda: self._set_all_checked(False))
        self._selection_label = QLabel("")
        select_row.addWidget(self._select_all_btn)
        select_row.addWidget(self._deselect_all_btn)
        select_row.addStretch()
        select_row.addWidget(self._selection_label)
        album_layout.addLayout(select_row)

        layout.addWidget(album_group, 1)

        # -- 下载控制 --
        ctrl_row = QHBoxLayout()
        self._download_btn = QPushButton("开始下载")
        self._download_btn.clicked.connect(self._on_start_download)
        self._download_btn.setEnabled(False)
        self._pause_btn = QPushButton("暂停")
        self._pause_btn.clicked.connect(self._on_pause)
        self._pause_btn.setEnabled(False)
        self._is_paused = False
        self._stop_btn = QPushButton("停止")
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        ctrl_row.addWidget(self._download_btn)
        ctrl_row.addWidget(self._pause_btn)
        ctrl_row.addWidget(self._stop_btn)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # -- 进度条 --
        self._progress = StatusProgressBar()
        layout.addWidget(self._progress)

        # -- 日志 --
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(150)
        log_layout.addWidget(self._log_text)
        layout.addWidget(log_group)

    # ── URL 历史 ─────────────────────────────────────────────────

    def _load_url_history(self) -> None:
        """从 JSON 文件加载 URL 历史记录到下拉框。"""
        try:
            if config.URL_HISTORY_FILE.exists():
                urls = json.loads(config.URL_HISTORY_FILE.read_text("utf-8"))
                if isinstance(urls, list):
                    for url in urls:
                        self._url_input.addItem(url)
        except Exception:
            pass  # 文件损坏不影响使用

    def _save_url_history(self, url: str) -> None:
        """保存 URL 到历史记录（最近使用的在最前面）。"""
        # 收集现有项
        urls = [self._url_input.itemText(i) for i in range(self._url_input.count())]

        # 去重，将当前 URL 移到最前
        if url in urls:
            urls.remove(url)
        urls.insert(0, url)

        # 限制数量
        urls = urls[: config.URL_HISTORY_MAX]

        # 更新下拉框
        self._url_input.blockSignals(True)
        self._url_input.clear()
        for u in urls:
            self._url_input.addItem(u)
        self._url_input.setCurrentIndex(0)
        self._url_input.blockSignals(False)

        # 持久化
        try:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            config.URL_HISTORY_FILE.write_text(
                json.dumps(urls, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    # ── 下载日志 (断点续传) ──────────────────────────────────────

    @staticmethod
    def _load_download_log() -> dict:
        """从 JSON 文件加载下载日志。"""
        try:
            if config.DOWNLOAD_LOG_FILE.exists():
                data = json.loads(config.DOWNLOAD_LOG_FILE.read_text("utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _save_download_log(self) -> None:
        """持久化下载日志到 JSON 文件。"""
        try:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            config.DOWNLOAD_LOG_FILE.write_text(
                json.dumps(self._download_log, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("保存下载日志失败: %s", e)

    def _log_key(self, album_id: str) -> str:
        """生成下载日志的键: {username}:{album_id}"""
        return f"{self._username}:{album_id}"

    # ── 辅助 ───────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        self._log_text.append(msg)
        self.status_message.emit(msg)

    def _get_selected_albums(self) -> list[Album]:
        selected = []
        for row in range(self._album_table.rowCount()):
            cb_widget = self._album_table.cellWidget(row, 0)
            if cb_widget:
                cb = cb_widget.findChild(QCheckBox)
                if cb and cb.isChecked():
                    selected.append(self._albums[row])
        return selected

    def _update_selection_label(self) -> None:
        selected = self._get_selected_albums()
        total_images = sum(a.image_count for a in selected)
        self._selection_label.setText(
            f"已选: {len(selected)} 个相册, {total_images} 张图片"
        )

    def _set_all_checked(self, checked: bool) -> None:
        for row in range(self._album_table.rowCount()):
            cb_widget = self._album_table.cellWidget(row, 0)
            if cb_widget:
                cb = cb_widget.findChild(QCheckBox)
                if cb:
                    cb.setChecked(checked)
        self._update_selection_label()

    def _set_ui_downloading(self, downloading: bool) -> None:
        self._fetch_btn.setEnabled(not downloading)
        self._download_btn.setEnabled(not downloading)
        self._category_combo.setEnabled(not downloading)
        self._pause_btn.setEnabled(downloading)
        self._stop_btn.setEnabled(downloading)
        if not downloading:
            self._is_paused = False
            self._pause_btn.setText("暂停")

    # ── 事件处理 ───────────────────────────────────────────────

    @Slot()
    def _on_browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择保存路径", self._dir_input.text())
        if path:
            self._dir_input.setText(path)

    @Slot()
    def _on_fetch_albums(self) -> None:
        url = self._url_input.currentText().strip()
        if not url:
            QMessageBox.warning(self, "提示", "请输入 Yupoo 相册地址")
            return

        try:
            self._username = parse_username(url)
        except ValueError as e:
            QMessageBox.warning(self, "URL 错误", str(e))
            return

        self._save_url_history(url)
        self._fetch_btn.setEnabled(False)
        self._log(f"正在获取分类列表 (用户: {self._username}) ...")

        # 取消旧的 fetch 任务
        if self._fetch_task and not self._fetch_task.done():
            self._fetch_task.cancel()
        self._fetch_task = asyncio.ensure_future(self._fetch_categories_async())

    async def _fetch_categories_async(self) -> None:
        """获取分类列表并填充 ComboBox，然后自动加载第一个分类的相册。"""
        try:
            async with aiohttp.ClientSession(connector=_create_connector()) as session:
                self._progress.set_status("正在获取分类列表 ...")
                self._categories = await get_categories(session, self._username)

            # 填充 ComboBox（先断开信号避免触发 _on_category_changed）
            self._category_combo.blockSignals(True)
            self._category_combo.clear()
            self._category_combo.addItem("[所有相册]", userData=None)
            for cat in self._categories:
                label = f"{cat.name} ({cat.album_count})" if cat.album_count else cat.name
                self._category_combo.addItem(label, userData=cat.category_id)
            self._category_combo.blockSignals(False)
            self._category_combo.setEnabled(True)

            if self._categories:
                self._log(f"获取到 {len(self._categories)} 个分类")
            else:
                self._log("该店铺无分类，将加载所有相册")

            # 自动加载当前选中分类的相册
            await self._fetch_albums_async()

        except asyncio.CancelledError:
            self._log("获取分类已取消")
            self._fetch_btn.setEnabled(True)
        except Exception as e:
            self._log(f"获取分类失败: {e}")
            QMessageBox.critical(self, "获取失败", str(e))
            self._fetch_btn.setEnabled(True)

    @Slot(int)
    def _on_category_changed(self, index: int) -> None:
        """分类切换时重新加载相册列表。"""
        if index < 0:
            return
        self._fetch_btn.setEnabled(False)
        self._download_btn.setEnabled(False)
        category_id = self._category_combo.currentData()
        name = self._category_combo.currentText()
        self._log(f"切换分类: {name}")
        if self._fetch_task and not self._fetch_task.done():
            self._fetch_task.cancel()
        self._fetch_task = asyncio.ensure_future(self._fetch_albums_async(category_id))

    async def _fetch_albums_async(self, category_id: str | None = None) -> None:
        """获取相册列表。category_id 为 None 时加载所有相册。"""
        if category_id is None:
            category_id = self._category_combo.currentData()

        try:
            async with aiohttp.ClientSession(connector=_create_connector()) as session:
                if category_id is not None:
                    self._progress.set_status("正在加载分类相册 ...")
                    self._albums = await get_category_albums(
                        session, self._username, category_id
                    )
                else:
                    def on_page(current, total):
                        self._progress.set_status(f"正在加载第 {current}/{total} 页")
                        self._progress.set_progress(current, total)

                    self._albums = await get_album_list(
                        session, self._username, on_progress=on_page
                    )

            self._populate_album_table()
            self._log(f"获取完成，共 {len(self._albums)} 个相册")

            # 去重检测
            await self._detect_duplicates()

            self._progress.set_status("就绪")
            self._progress.set_progress(0, 0)
            self._download_btn.setEnabled(True)

        except asyncio.CancelledError:
            self._log("获取相册列表已取消")
        except Exception as e:
            self._log(f"获取相册列表失败: {e}")
            QMessageBox.critical(self, "获取失败", str(e))

        finally:
            self._fetch_btn.setEnabled(True)

    def _populate_album_table(self) -> None:
        self._album_table.setRowCount(len(self._albums))
        for row, album in enumerate(self._albums):
            # 查询下载日志
            log_entry = self._download_log.get(self._log_key(album.album_id))
            is_done = log_entry and log_entry.get("status") == "done"
            is_partial = log_entry and log_entry.get("status") == "partial"

            # 勾选框
            cb = QCheckBox()
            cb.setChecked(not is_done)  # done → 取消勾选, partial/new → 勾选
            cb.stateChanged.connect(lambda _: self._update_selection_label())
            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self._album_table.setCellWidget(row, 0, cb_widget)

            # 名称
            name_item = QTableWidgetItem(album.title)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._album_table.setItem(row, 1, name_item)

            # 数量
            count_item = QTableWidgetItem(str(album.image_count))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._album_table.setItem(row, 2, count_item)

            # 状态
            status_text = ""
            if is_done:
                status_text = "✓ 已下载"
            elif is_partial:
                dl = log_entry.get("downloaded", 0)
                total = log_entry.get("image_count", 0)
                status_text = f"部分 ({dl}/{total})"

            status_item = QTableWidgetItem(status_text)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._album_table.setItem(row, 3, status_item)

            # done → 行变灰
            if is_done:
                grey_bg = QBrush(QColor(230, 230, 230))
                for col in range(self._album_table.columnCount()):
                    item = self._album_table.item(row, col)
                    if item:
                        item.setBackground(grey_bg)

        self._update_selection_label()

    # ── 相册去重 ─────────────────────────────────────────────────

    def _mark_row_duplicate(self, row: int, reason: str) -> None:
        """将指定行标记为重复：取消勾选、背景变黄、写入状态。"""
        # 取消勾选
        cb_widget = self._album_table.cellWidget(row, 0)
        if cb_widget:
            cb = cb_widget.findChild(QCheckBox)
            if cb:
                cb.setChecked(False)

        # 黄色背景
        bg = QBrush(QColor(255, 255, 200))
        for col in range(self._album_table.columnCount()):
            item = self._album_table.item(row, col)
            if item:
                item.setBackground(bg)

        # 状态文字
        status_item = self._album_table.item(row, 3)
        if status_item:
            status_item.setText(reason)

    async def _detect_duplicates(self) -> None:
        """检测相册列表中的重复项（标题相同 + 封面 pHash）。

        先做标题精确匹配（零成本），再异步下载封面图做 pHash 对比。
        重复项自动取消勾选并标黄，用户可手动重新勾选。
        """
        if len(self._albums) < 2:
            return

        dup_count = 0

        # ── 第一层：标题精确匹配 ──
        title_first_seen: dict[str, int] = {}  # title → 首次出现的行号
        for row, album in enumerate(self._albums):
            if album.title in title_first_seen:
                self._mark_row_duplicate(row, f"标题重复(={title_first_seen[album.title]+1}行)")
                dup_count += 1
            else:
                title_first_seen[album.title] = row

        if dup_count:
            self._log(f"标题去重: 发现 {dup_count} 个重复相册")

        # ── 第二层：封面 pHash ──
        # 收集有封面 URL 且尚未被标记的相册
        cover_tasks: list[tuple[int, str]] = []  # (row, cover_url)
        for row, album in enumerate(self._albums):
            status = self._album_table.item(row, 3)
            if status and status.text():
                continue  # 已被标题去重标记
            if album.cover_url:
                cover_tasks.append((row, album.cover_url))

        if len(cover_tasks) < 2:
            self._update_selection_label()
            return

        self._progress.set_status("正在检测封面重复 ...")
        phash_dup_count = 0

        # 下载封面并计算 pHash
        hashes: list[tuple[int, imagehash.ImageHash | None]] = []
        sem = asyncio.Semaphore(config.COVER_DOWNLOAD_CONCURRENCY)

        async def _fetch_cover_hash(
            session: aiohttp.ClientSession,
            row: int,
            url: str,
        ) -> tuple[int, imagehash.ImageHash | None]:
            async with sem:
                try:
                    headers = {
                        "User-Agent": config.HEADERS["User-Agent"],
                        "Referer": config.REFERER_TEMPLATE.format(username=self._username),
                    }
                    timeout = aiohttp.ClientTimeout(total=10)
                    async with session.get(url, headers=headers, timeout=timeout) as resp:
                        if resp.status != 200:
                            return row, None
                        data = await resp.read()
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                        h = imagehash.phash(img, hash_size=config.COVER_PHASH_SIZE)
                        return row, h
                except Exception as e:
                    logger.debug("封面下载失败 row=%d: %s", row, e)
                    return row, None

        async with aiohttp.ClientSession(connector=_create_connector()) as session:
            tasks = [
                _fetch_cover_hash(session, row, url)
                for row, url in cover_tasks
            ]
            hashes = await asyncio.gather(*tasks)

        # 比较 pHash
        valid_hashes: list[tuple[int, imagehash.ImageHash]] = [
            (row, h) for row, h in hashes if h is not None
        ]

        seen_phashes: list[tuple[int, imagehash.ImageHash]] = []
        for row, h in valid_hashes:
            # 检查该行是否已被标记（防止标题去重后又被 pHash 重复标记）
            status = self._album_table.item(row, 3)
            if status and status.text():
                seen_phashes.append((row, h))
                continue

            matched_row = None
            for prev_row, prev_h in seen_phashes:
                if abs(h - prev_h) <= config.COVER_PHASH_THRESHOLD:
                    matched_row = prev_row
                    break

            if matched_row is not None:
                self._mark_row_duplicate(row, f"封面重复(≈{matched_row+1}行)")
                phash_dup_count += 1
            else:
                seen_phashes.append((row, h))

        if phash_dup_count:
            self._log(f"封面去重: 发现 {phash_dup_count} 个重复相册")

        total_dup = dup_count + phash_dup_count
        if total_dup:
            self._log(f"共检测到 {total_dup} 个疑似重复相册（已自动取消勾选，可手动恢复）")
        else:
            self._log("未检测到重复相册")

        self._update_selection_label()

    # ── 下载流程 ───────────────────────────────────────────────

    @Slot()
    def _on_start_download(self) -> None:
        selected = self._get_selected_albums()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一个相册")
            return

        download_dir = Path(self._dir_input.text().strip())
        if not download_dir.exists():
            download_dir.mkdir(parents=True, exist_ok=True)

        self._downloader = Downloader(
            username=self._username,
            download_dir=download_dir,
        )

        self._set_ui_downloading(True)
        self._download_task = asyncio.ensure_future(
            self._download_flow(selected, download_dir)
        )

    async def _download_flow(self, albums: list[Album], download_dir: Path) -> None:
        """生产者-消费者模式：边扫描相册边下载图片。

        生产者: 逐个获取相册的图片 URL，放入队列
        消费者: 从队列取出相册，立即下载图片
        两者并行，总耗时更短，用户几秒后就能看到文件落盘。
        """
        producer_task = None
        consumer_task = None

        try:
            # maxsize=2: 生产者最多领先消费者 2 个相册，控制内存
            queue: asyncio.Queue[tuple[AlbumDetail, str] | None] = asyncio.Queue(maxsize=2)
            total_albums = len(albums)
            scanned_albums = 0
            total_images = 0
            downloaded_images = 0

            async with aiohttp.ClientSession(connector=_create_connector()) as session:

                async def producer():
                    nonlocal scanned_albums, total_images
                    try:
                        for i, album in enumerate(albums):
                            if self._downloader.is_cancelled:
                                break

                            # 相册间延迟，避免触发 429
                            if i > 0:
                                await asyncio.sleep(random.uniform(*config.REQUEST_DELAY))

                            self._progress.set_status(
                                f"扫描相册 ({i + 1}/{total_albums})"
                            )

                            detail = await get_album_images(session, album, self._username)

                            # 断点续传：优先复用日志中记录的文件夹名
                            log_key = self._log_key(album.album_id)
                            log_entry = self._download_log.get(log_key)
                            if log_entry and (download_dir / log_entry["folder"]).exists():
                                folder_name = log_entry["folder"]
                            else:
                                _, folder_name = unique_folder_path(download_dir, album.title)

                            scanned_albums += 1
                            total_images += len(detail.image_urls)

                            self._log(
                                f"  [{scanned_albums}/{total_albums}] {album.title}: "
                                f"{len(detail.image_urls)} 张图片"
                            )

                            await queue.put((detail, folder_name))
                    finally:
                        await queue.put(None)  # 结束标记，确保消费者退出

                async def consumer():
                    nonlocal downloaded_images
                    result = BatchResult()

                    while True:
                        item = await queue.get()
                        if item is None:
                            break
                        if self._downloader.is_cancelled:
                            break

                        detail, folder_name = item
                        self._log(f"▶ 下载相册: {detail.album.title}")

                        def on_progress(completed, total, dl_result):
                            nonlocal downloaded_images
                            downloaded_images += 1
                            status = f"下载中: {dl_result.path.name}"
                            if scanned_albums < total_albums:
                                status += f" (扫描 {scanned_albums}/{total_albums})"
                            self._progress.set_status(status)
                            self._progress.set_progress(downloaded_images, total_images)

                        album_result = await self._downloader.download_album(
                            session, detail, folder_name, on_progress=on_progress,
                        )

                        result.total += album_result.total
                        result.success += album_result.success
                        result.skipped += album_result.skipped
                        result.failed += album_result.failed
                        result.errors.extend(album_result.errors)

                        self._log(
                            f"  ✓ {detail.album.title}: "
                            f"成功 {album_result.success}, 跳过 {album_result.skipped}, "
                            f"失败 {album_result.failed}"
                        )

                        # 更新下载日志
                        album = detail.album
                        log_key = self._log_key(album.album_id)
                        downloaded_count = album_result.success + album_result.skipped
                        self._download_log[log_key] = {
                            "album_id": album.album_id,
                            "username": self._username,
                            "title": album.title,
                            "folder": folder_name,
                            "image_count": album_result.total,
                            "downloaded": downloaded_count,
                            "status": "done" if album_result.failed == 0 else "partial",
                            "updated_at": datetime.now().isoformat(timespec="seconds"),
                        }
                        self._save_download_log()

                    return result

                self._log(f"开始采集: {total_albums} 个相册 (边扫描边下载)")

                producer_task = asyncio.create_task(producer())
                consumer_task = asyncio.create_task(consumer())

                await asyncio.gather(producer_task, consumer_task)
                result = consumer_task.result()

            # 完成
            self._log(
                f"\n下载完成! "
                f"成功: {result.success}, 跳过: {result.skipped}, "
                f"失败: {result.failed}, 总计: {result.total}"
            )
            if result.errors:
                self._log("错误详情:")
                for err in result.errors:
                    self._log(f"  {err}")

            self._progress.set_status("下载完成")

        except asyncio.CancelledError:
            self._log("下载已取消")
        except Exception as e:
            self._log(f"下载出错: {e}")
            QMessageBox.critical(self, "下载失败", str(e))
        finally:
            # 确保子任务被清理
            for task in (producer_task, consumer_task):
                if task and not task.done():
                    task.cancel()
            self._set_ui_downloading(False)

    @Slot()
    def _on_pause(self) -> None:
        if self._downloader is None:
            return
        if not self._is_paused:
            self._downloader.pause()
            self._is_paused = True
            self._pause_btn.setText("继续")
            self._progress.set_status("已暂停")
            self._log("下载已暂停")
        else:
            self._downloader.resume()
            self._is_paused = False
            self._pause_btn.setText("暂停")
            self._progress.set_status("下载中")
            self._log("下载已恢复")

    @Slot()
    def _on_stop(self) -> None:
        if self._downloader:
            self._downloader.cancel()
            self._log("正在停止 ...")
        if self._download_task and not self._download_task.done():
            self._download_task.cancel()
        # 如果 5 秒后仍未停止，提示用户
        from PySide6.QtCore import QTimer
        QTimer.singleShot(5000, self._check_stop_timeout)

    def _check_stop_timeout(self) -> None:
        """检查停止操作是否超时。"""
        if self._download_task and not self._download_task.done():
            self._progress.set_status("停止超时，请稍候或重启应用")

    def cleanup(self) -> None:
        """应用关闭时调用，取消进行中的下载和获取。"""
        if self._downloader:
            self._downloader.cancel()
        if self._download_task and not self._download_task.done():
            self._download_task.cancel()
        if self._fetch_task and not self._fetch_task.done():
            self._fetch_task.cancel()
