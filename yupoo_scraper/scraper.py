"""Yupoo 页面解析 — 提取相册列表和图片 URL"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urljoin, parse_qs, urlparse

import aiohttp
from bs4 import BeautifulSoup

from . import config

logger = logging.getLogger(__name__)


@dataclass
class Album:
    """一个 Yupoo 相册"""
    album_id: str
    title: str
    url: str               # 相册详情页完整 URL
    image_count: int = 0   # 相册列表页显示的图片数量
    cover_url: str = ""    # 封面缩略图 URL


@dataclass
class AlbumDetail:
    """相册详情，包含所有图片 URL"""
    album: Album
    image_urls: list[str] = field(default_factory=list)


@dataclass
class Category:
    """Yupoo 店铺分类"""
    category_id: str
    name: str
    album_count: int = 0


def parse_username(url: str) -> str:
    """从 Yupoo URL 中提取用户名。

    支持格式:
      - https://x.yupoo.com/photos/{username}/albums
      - https://{username}.x.yupoo.com/albums
    """
    m = re.search(r"x\.yupoo\.com/photos/([^/]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"//([^.]+)\.x\.yupoo\.com", url)
    if m:
        return m.group(1)
    raise ValueError(f"无法从 URL 中提取用户名: {url}")


def _build_base_url(username: str) -> str:
    return f"https://x.yupoo.com/photos/{username}"


def _make_headers(username: str) -> dict[str, str]:
    return config.make_headers(username)


# ── 解析相册列表页 HTML ─────────────────────────────────────────

def _parse_albums_page(html: str, base_url: str) -> list[Album]:
    """从一页 HTML 中提取所有相册信息。

    支持两种页面布局:
      - 列表视图 (albums?page=N): 选择器 a.album3__main
      - 画廊视图 (albums?tab=gallery&referrercate=...): 选择器 a.album__main
    """
    soup = BeautifulSoup(html, "lxml")
    albums: list[Album] = []

    # 列表视图优先，无结果时尝试画廊视图
    a_tags = soup.select(config.SEL_ALBUM_LINK)
    if not a_tags:
        a_tags = soup.select(config.SEL_ALBUM_LINK_GALLERY)

    for a_tag in a_tags:
        # 相册 ID：优先用 data-album-id 属性，其次从 href 提取
        album_id = a_tag.get("data-album-id", "")
        href = a_tag.get("href", "")
        if not album_id:
            id_match = re.search(r"/albums/(\d+)", href)
            album_id = id_match.group(1) if id_match else ""

        # 标题：优先用 title 属性，其次用子元素文本
        title = a_tag.get("title", "").strip()
        if not title:
            title_div = (
                a_tag.select_one(config.SEL_ALBUM_TITLE)
                or a_tag.select_one(config.SEL_ALBUM_TITLE_GALLERY)
            )
            title = title_div.get_text(strip=True) if title_div else album_id

        # 相册 URL
        album_url = urljoin(base_url + "/", href)

        # 图片数量
        count_div = a_tag.select_one("div.album__photonumber")
        if not count_div:
            count_div = a_tag.select_one("div.album3__photonumber")
        image_count = 0
        if count_div:
            try:
                image_count = int(count_div.get_text(strip=True))
            except ValueError:
                pass

        # 封面
        cover_img = (
            a_tag.select_one("img.album__img")
            or a_tag.select_one("img.album3__img")
        )
        cover_url = ""
        if cover_img:
            cover_url = (
                cover_img.get("data-origin-src", "")
                or cover_img.get("data-src", "")
                or cover_img.get("src", "")
            )
            if cover_url and not cover_url.startswith("http"):
                cover_url = "https:" + cover_url

        albums.append(Album(
            album_id=album_id,
            title=title,
            url=album_url,
            image_count=image_count,
            cover_url=cover_url,
        ))

    return albums


def _parse_max_page(html: str) -> int:
    """从分页控件中提取最大页码，无分页则返回 1。"""
    soup = BeautifulSoup(html, "lxml")
    page_input = soup.select_one(config.SEL_PAGINATION_MAX)
    if page_input and page_input.get("max"):
        try:
            return int(page_input["max"])
        except (ValueError, TypeError):
            return 1
    return 1


# ── 解析单个相册页 HTML ─────────────────────────────────────────

def _parse_album_images(html: str) -> list[str]:
    """从相册详情页提取所有图片 URL (big 尺寸)。"""
    soup = BeautifulSoup(html, "lxml")
    urls: list[str] = []

    for container in soup.select(config.SEL_IMAGE_CONTAINER):
        img = container.select_one("img")
        if not img:
            continue
        url = img.get("data-src", "")
        if not url or not url.startswith("http"):
            continue
        # 确保是 big 尺寸
        url = _ensure_size(url, config.IMAGE_SIZE)
        urls.append(url)

    return urls


def _ensure_size(url: str, size: str) -> str:
    """将图片 URL 中的尺寸替换为指定尺寸。

    例: .../hash/small.jpeg → .../hash/big.jpeg
    """
    return re.sub(
        r"/(square|small|medium|big|large|original|full)\.",
        f"/{size}.",
        url,
    )


# ── 带重试的请求 ───────────────────────────────────────────────

# 值得自动重试的 HTTP 状态码（服务端临时故障）
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


async def _fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    *,
    max_retries: int = config.MAX_RETRIES,
) -> str:
    """GET 请求，遇到 429/5xx 自动退避重试，返回响应文本。"""
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            async with session.get(
                url, headers=headers,
                timeout=aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT),
            ) as resp:
                if resp.status in _RETRYABLE_STATUS:
                    if attempt < max_retries:
                        wait = config.retry_wait(attempt, dict(resp.headers))
                        logger.warning(
                            "HTTP %d on %s (attempt %d/%d), retrying in %.1fs",
                            resp.status, url, attempt, max_retries, wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    # 最后一次仍然失败，抛出
                    resp.raise_for_status()
                resp.raise_for_status()
                return await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = e
            if attempt < max_retries:
                wait = config.retry_wait(attempt)
                logger.warning(
                    "%s on %s (attempt %d/%d), retrying in %.1fs",
                    type(e).__name__, url, attempt, max_retries, wait,
                )
                await asyncio.sleep(wait)

    # 所有重试耗尽
    raise last_error or aiohttp.ClientError(
        f"请求失败，已重试 {max_retries} 次: {url}"
    )


# ── 异步 API ───────────────────────────────────────────────────

async def get_album_list(
    session: aiohttp.ClientSession,
    username: str,
    *,
    on_progress: config.ProgressCallback | None = None,
) -> list[Album]:
    """获取所有相册列表（自动处理分页）。

    使用 /categories?page=N 端点代替 /albums?page=N，
    因为后者可能遗漏部分相册（例如 554 个只返回 440 个）。

    Args:
        session: aiohttp 会话
        username: Yupoo 用户名
        on_progress: 可选回调 (current_page, total_pages)

    Returns:
        所有相册的列表
    """
    base_url = _build_base_url(username)
    headers = _make_headers(username)
    all_albums: list[Album] = []
    seen_ids: set[str] = set()

    # 第一页 — 确定总页数
    page1_url = f"{base_url}/categories?page=1"
    html = await _fetch_with_retry(session, page1_url, headers)

    max_page = _parse_max_page(html)
    for album in _parse_albums_page(html, base_url):
        if album.album_id and album.album_id not in seen_ids:
            seen_ids.add(album.album_id)
            all_albums.append(album)

    if on_progress:
        on_progress(1, max_page)

    # 后续页
    for page_num in range(2, max_page + 1):
        # 页面间延迟，避免触发 429
        await asyncio.sleep(random.uniform(*config.REQUEST_DELAY))

        page_url = f"{base_url}/categories?page={page_num}"
        html = await _fetch_with_retry(session, page_url, headers)

        for album in _parse_albums_page(html, base_url):
            if album.album_id and album.album_id not in seen_ids:
                seen_ids.add(album.album_id)
                all_albums.append(album)

        if on_progress:
            on_progress(page_num, max_page)

    return all_albums


async def get_categories(
    session: aiohttp.ClientSession,
    username: str,
) -> list[Category]:
    """获取店铺的分类列表。

    从相册首页侧边栏 ul.showheader__category 解析分类链接。

    Returns:
        分类列表（不含"所有相册"，由调用方添加）
    """
    base_url = _build_base_url(username)
    headers = _make_headers(username)

    html = await _fetch_with_retry(session, f"{base_url}/albums", headers)
    soup = BeautifulSoup(html, "lxml")

    categories: list[Category] = []
    for a_tag in soup.select(config.SEL_CATEGORY_LINK):
        href = a_tag.get("href", "")
        name = a_tag.get_text(strip=True)
        if not name:
            continue

        # 提取分类 ID：
        # 格式1: /photos/{user}/categories/{id} (路径中)
        # 格式2: ?referrercate={id} 或 ?cate={id} (查询参数中)
        path_match = re.search(r"/categories/(\d+)", href)
        if path_match:
            category_id = path_match.group(1)
        else:
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            cate_values = qs.get("referrercate") or qs.get("cate")
            if not cate_values:
                continue
            category_id = cate_values[0]

        # 尝试从文本中提取相册数量，格式可能是 "分类名(12)"
        album_count = 0
        count_match = re.search(r"\((\d+)\)$", name)
        if count_match:
            album_count = int(count_match.group(1))
            name = name[:count_match.start()].strip()

        categories.append(Category(
            category_id=category_id,
            name=name,
            album_count=album_count,
        ))

    return categories


async def get_category_albums(
    session: aiohttp.ClientSession,
    username: str,
    category_id: str,
) -> list[Album]:
    """获取某个分类下的所有相册。

    使用 /albums?tab=gallery&referrercate={category_id}，
    此接口一页返回所有相册，无需分页。
    """
    base_url = _build_base_url(username)
    headers = _make_headers(username)

    url = config.CATEGORY_ALBUMS_URL.format(
        username=username, category_id=category_id
    )
    html = await _fetch_with_retry(session, url, headers)
    return _parse_albums_page(html, base_url)


async def get_album_images(
    session: aiohttp.ClientSession,
    album: Album,
    username: str,
) -> AlbumDetail:
    """获取单个相册内所有图片 URL。"""
    headers = _make_headers(username)
    html = await _fetch_with_retry(session, album.url, headers)
    image_urls = _parse_album_images(html)
    return AlbumDetail(album=album, image_urls=image_urls)
