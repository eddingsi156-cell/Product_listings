"""异步下载管理器 — 并发控制、重试、断点续传"""

from __future__ import annotations

import asyncio
import logging
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import aiohttp
import yarl

logger = logging.getLogger(__name__)

from . import config
from .scraper import AlbumDetail


@dataclass
class DownloadResult:
    """单张图片下载结果"""
    url: str
    path: Path
    success: bool
    skipped: bool = False  # 文件已存在，跳过
    error: str = ""


@dataclass
class BatchResult:
    """批量下载汇总"""
    total: int = 0
    success: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] | None = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


# 常见图片格式的 magic bytes
_IMAGE_MAGIC = (
    b'\xff\xd8\xff',     # JPEG
    b'\x89PNG\r\n',      # PNG
    b'RIFF',             # WebP (RIFF....WEBP)
    b'GIF87a', b'GIF89a',  # GIF
    b'BM',               # BMP
)


def _is_valid_image_file(path: Path) -> bool:
    """通过 magic bytes 检查文件是否为有效图片（排除 CDN 挑战 HTML）。"""
    try:
        with open(path, 'rb') as f:
            header = f.read(12)
        if len(header) < 4:
            return False
        return header.startswith(_IMAGE_MAGIC)
    except OSError:
        return False


def _solve_cdn_challenge(data: bytes) -> dict[str, str] | None:
    """解析 TencentEdgeOne CDN 的 JS cookie 挑战。

    CDN 返回一段混淆 JS，设置两个 cookie 后重定向:
      - __tst_status = sum(三个大整数) + "#"
      - EO_Bot_Ssid = 第一个大整数

    Returns:
        解析出的 cookie 字典，或 None（非挑战响应）。
    """
    # 快速判断：挑战页面很小（< 2KB）且包含特征字符串
    if len(data) > 2048:
        return None

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return None

    if "__tst_status" not in text or "EO_Bot_Ssid" not in text:
        return None

    # 提取所有 9 位以上整数（CDN 挑战中恰好有 4 个）
    nums = [int(n) for n in re.findall(r"\b(\d{9,})\b", text)]
    if len(nums) < 4:
        return None

    # 第一个大数是 EO_Bot_Ssid，后三个求和为 __tst_status
    eo_bot_ssid = nums[0]
    tst_status = sum(nums[1:4])

    return {
        "__tst_status": f"{tst_status}#",
        "EO_Bot_Ssid": str(eo_bot_ssid),
    }


class Downloader:
    """异步图片下载器。

    用法:
        downloader = Downloader(username="gyg88", download_dir=Path("./downloads"))
        result = await downloader.download_album(album_detail, on_progress=callback)
    """

    def __init__(
        self,
        username: str,
        download_dir: Path,
        *,
        max_concurrency: int = config.MAX_CONCURRENCY,
        delay_range: tuple[float, float] = config.REQUEST_DELAY,
        max_retries: int = config.MAX_RETRIES,
    ):
        self.username = username
        self.download_dir = download_dir
        self.max_concurrency = max_concurrency
        self.delay_range = delay_range
        self.max_retries = max_retries

        self._semaphore: asyncio.Semaphore | None = None
        self._cancel_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # 初始非暂停
        self._challenge_solved: set[str] = set()  # 已解决挑战的 CDN 域名
        self._active_tasks: list[asyncio.Task] = []  # 当前活跃的下载任务

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def cancel(self):
        """取消下载（立即取消所有活跃任务）"""
        self._cancel_event.set()
        for task in self._active_tasks:
            task.cancel()

    def pause(self):
        """暂停下载"""
        self._pause_event.clear()

    def resume(self):
        """恢复下载"""
        self._pause_event.set()

    def reset(self):
        """重置状态以用于新一轮下载（在当前事件循环中重建 Semaphore）。"""
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._cancel_event.clear()
        self._pause_event.set()

    def _make_headers(self) -> dict[str, str]:
        return config.make_headers(self.username)

    async def _solve_challenge_if_needed(
        self,
        session: aiohttp.ClientSession,
        url: str,
        data: bytes,
    ) -> bool:
        """检测并解决 CDN anti-bot 挑战，成功返回 True。"""
        cookies = _solve_cdn_challenge(data)
        if cookies is None:
            return False

        # 将 cookie 设置到对应的 CDN 域名
        parsed = urlparse(url)
        host = parsed.hostname or ""
        cookie_url = f"https://{host}/"

        for name, value in cookies.items():
            session.cookie_jar.update_cookies(
                {name: value},
                response_url=yarl.URL(cookie_url),
            )

        self._challenge_solved.add(host)
        return True

    async def download_image(
        self,
        session: aiohttp.ClientSession,
        url: str,
        save_path: Path,
    ) -> DownloadResult:
        """下载单张图片，带重试和指数退避。"""
        # 断点续传：文件已存在且是有效图片则跳过
        # 不能仅靠文件大小判断——CDN 挑战页面约 988 bytes，会被误判为有效文件
        if save_path.exists() and _is_valid_image_file(save_path):
            return DownloadResult(url=url, path=save_path, success=True, skipped=True)

        headers = self._make_headers()
        last_error = ""

        for attempt in range(1, self.max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT)
                async with session.get(url, headers=headers, timeout=timeout) as resp:
                    if resp.status == 567:
                        last_error = "HTTP 567: Referer 校验失败"
                        break  # Referer 问题不重试
                    if resp.status == 429:
                        last_error = "HTTP 429: 请求过于频繁"
                        if attempt < self.max_retries:
                            wait = config.retry_wait(attempt, dict(resp.headers))
                            await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()

                    data = await resp.read()

                    # 检测 CDN anti-bot 挑战
                    if await self._solve_challenge_if_needed(session, url, data):
                        # 挑战已解决，立即重试（不计入重试次数）
                        await asyncio.sleep(1.5)
                        async with session.get(url, headers=headers, timeout=timeout) as resp2:
                            resp2.raise_for_status()
                            data = await resp2.read()
                            # 二次检查（防止无限循环）
                            if _solve_cdn_challenge(data) is not None:
                                last_error = "CDN 挑战解决失败（cookie 未被接受）"
                                continue

                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    save_path.write_bytes(data)

                return DownloadResult(url=url, path=save_path, success=True)

            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries:
                    await asyncio.sleep(config.retry_wait(attempt))

        return DownloadResult(url=url, path=save_path, success=False, error=last_error)

    async def _warm_cdn_cookies(self, session: aiohttp.ClientSession, urls: list[str]):
        """预先解决 CDN 挑战，避免并发下载时多次遇到挑战。"""
        # 收集需要预热的 CDN 域名
        domains_to_warm: set[str] = set()
        for url in urls:
            host = urlparse(url).hostname or ""
            if host not in self._challenge_solved:
                domains_to_warm.add(host)

        if not domains_to_warm:
            return

        headers = self._make_headers()
        timeout = aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT)

        for host in domains_to_warm:
            # 找一个该域名的 URL 做探测
            probe_url = next((u for u in urls if host in u), None)
            if not probe_url:
                continue

            try:
                async with session.get(probe_url, headers=headers, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        if await self._solve_challenge_if_needed(session, probe_url, data):
                            await asyncio.sleep(1.5)
            except Exception as e:
                logger.debug("CDN cookie 预热失败 (%s): %s", host, e)

    async def download_album(
        self,
        session: aiohttp.ClientSession,
        album_detail: AlbumDetail,
        folder_name: str,
        *,
        on_progress: Callable[[int, int, DownloadResult], None] | None = None,
    ) -> BatchResult:
        """下载一个相册的所有图片。

        Args:
            session: aiohttp 会话
            album_detail: 相册详情（含图片 URL 列表）
            folder_name: 保存文件夹名（已清理过非法字符）
            on_progress: 可选回调 (completed, total, result)
        """
        urls = album_detail.image_urls

        # 防御路径穿越：确保目标目录在 download_dir 内
        album_dir = (self.download_dir / folder_name).resolve()
        if not album_dir.is_relative_to(self.download_dir.resolve()):
            raise ValueError(f"非法文件夹名（路径穿越）: {folder_name!r}")

        # 预热 CDN cookie，解决 anti-bot 挑战
        await self._warm_cdn_cookies(session, urls)
        album_dir.mkdir(parents=True, exist_ok=True)

        batch = BatchResult(total=len(urls))
        completed = 0

        async def _download_one(idx: int, url: str):
            nonlocal completed

            # 检查取消
            if self._cancel_event.is_set():
                return

            # 等待暂停恢复
            await self._pause_event.wait()

            # 确定文件扩展名
            ext = _extract_ext(url)
            save_path = album_dir / f"{idx + 1:02d}{ext}"

            if self._semaphore is None:
                self._semaphore = asyncio.Semaphore(self.max_concurrency)
            async with self._semaphore:
                if self._cancel_event.is_set():
                    return

                result = await self.download_image(session, url, save_path)

                # 请求间延迟
                if not result.skipped:
                    delay = random.uniform(*self.delay_range)
                    await asyncio.sleep(delay)

            # 更新统计
            if result.success:
                if result.skipped:
                    batch.skipped += 1
                else:
                    batch.success += 1
            else:
                batch.failed += 1
                batch.errors.append(f"[{folder_name}/{save_path.name}] {result.error}")

            completed += 1
            if on_progress:
                on_progress(completed, batch.total, result)

        self._active_tasks = [
            asyncio.ensure_future(_download_one(i, url))
            for i, url in enumerate(urls)
        ]
        try:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        finally:
            self._active_tasks.clear()

        return batch

    async def download_albums(
        self,
        session: aiohttp.ClientSession,
        album_details: list[tuple[AlbumDetail, str]],
        *,
        on_album_start: Callable | None = None,
        on_image_progress: Callable | None = None,
        on_album_done: Callable | None = None,
    ) -> BatchResult:
        """下载多个相册。

        Args:
            album_details: [(AlbumDetail, folder_name), ...]
            on_album_start: 回调 (album_index, album_title)
            on_image_progress: 回调 (completed, total, result)
            on_album_done: 回调 (album_index, album_title, batch_result)
        """
        total_result = BatchResult()

        for idx, (detail, folder_name) in enumerate(album_details):
            if self._cancel_event.is_set():
                break

            if on_album_start:
                on_album_start(idx, detail.album.title)

            result = await self.download_album(
                session, detail, folder_name,
                on_progress=on_image_progress,
            )

            total_result.total += result.total
            total_result.success += result.success
            total_result.skipped += result.skipped
            total_result.failed += result.failed
            total_result.errors.extend(result.errors)

            if on_album_done:
                on_album_done(idx, detail.album.title, result)

        return total_result


def _extract_ext(url: str) -> str:
    """从 URL 中提取文件扩展名。"""
    # e.g. https://photo.yupoo.com/gyg88/hash/big.jpeg → .jpeg
    path = url.split("?")[0]
    dot_pos = path.rfind(".")
    if dot_pos != -1:
        ext = path[dot_pos:]
        if len(ext) <= 5:  # .jpeg, .jpg, .png, .webp
            return ext
    return ".jpeg"
