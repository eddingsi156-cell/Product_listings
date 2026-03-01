"""scraper 模块单元测试 — HTML 解析"""

import pytest

from yupoo_scraper.scraper import (
    _parse_albums_page,
    _parse_album_images,
    _parse_max_page,
)


# ── _parse_albums_page ───────────────────────────────────────

SAMPLE_LIST_VIEW = """
<html>
<body>
<a class="album3__main" href="/photos/user1/albums/123456" data-album-id="123456" title="Nike Air Max">
    <div class="album3__title">Nike Air Max</div>
    <div class="album3__photonumber">12</div>
    <img class="album3__img" data-origin-src="//photo.yupoo.com/cover1.jpg">
</a>
<a class="album3__main" href="/photos/user1/albums/789012" data-album-id="789012" title="Adidas Ultra Boost">
    <div class="album3__title">Adidas Ultra Boost</div>
    <div class="album3__photonumber">8</div>
</a>
</body>
</html>
"""

SAMPLE_GALLERY_VIEW = """
<html>
<body>
<a class="album__main" href="/photos/user1/albums/111222">
    <div class="album__title">Gallery Item</div>
    <div class="album__photonumber">5</div>
    <img class="album__img" src="//photo.yupoo.com/gallery_cover.jpg">
</a>
</body>
</html>
"""


class TestParseAlbumsPage:
    def test_list_view_basic(self):
        base = "https://x.yupoo.com/photos/user1"
        albums = _parse_albums_page(SAMPLE_LIST_VIEW, base)
        assert len(albums) == 2
        assert albums[0].album_id == "123456"
        assert albums[0].title == "Nike Air Max"
        assert albums[0].image_count == 12
        assert albums[1].album_id == "789012"

    def test_list_view_url_resolved(self):
        base = "https://x.yupoo.com/photos/user1"
        albums = _parse_albums_page(SAMPLE_LIST_VIEW, base)
        assert "albums/123456" in albums[0].url

    def test_list_view_cover_url(self):
        base = "https://x.yupoo.com/photos/user1"
        albums = _parse_albums_page(SAMPLE_LIST_VIEW, base)
        assert albums[0].cover_url.startswith("https:")
        assert "cover1.jpg" in albums[0].cover_url

    def test_gallery_view_fallback(self):
        base = "https://x.yupoo.com/photos/user1"
        albums = _parse_albums_page(SAMPLE_GALLERY_VIEW, base)
        assert len(albums) == 1
        assert albums[0].title == "Gallery Item"

    def test_gallery_view_id_from_href(self):
        base = "https://x.yupoo.com/photos/user1"
        albums = _parse_albums_page(SAMPLE_GALLERY_VIEW, base)
        assert albums[0].album_id == "111222"

    def test_empty_html(self):
        albums = _parse_albums_page("<html></html>", "https://x.yupoo.com/photos/user1")
        assert albums == []


# ── _parse_album_images ──────────────────────────────────────

SAMPLE_ALBUM_PAGE = """
<html>
<body>
<div class="showalbum__children">
    <img data-src="https://photo.yupoo.com/hash1/small.jpeg">
</div>
<div class="showalbum__children">
    <img data-src="https://photo.yupoo.com/hash2/medium.png">
</div>
<div class="showalbum__children">
    <img data-src="">
</div>
<div class="showalbum__children">
    <!-- 无 img 标签 -->
</div>
</body>
</html>
"""


class TestParseAlbumImages:
    def test_extracts_image_urls(self):
        urls = _parse_album_images(SAMPLE_ALBUM_PAGE)
        assert len(urls) == 2

    def test_converts_to_big_size(self):
        urls = _parse_album_images(SAMPLE_ALBUM_PAGE)
        assert "/big.jpeg" in urls[0]
        assert "/big.png" in urls[1]

    def test_skips_empty_data_src(self):
        urls = _parse_album_images(SAMPLE_ALBUM_PAGE)
        assert len(urls) == 2  # 空 data-src 和无 img 的不算

    def test_empty_page(self):
        urls = _parse_album_images("<html></html>")
        assert urls == []


# ── _parse_max_page ──────────────────────────────────────────


class TestParseMaxPage:
    def test_with_pagination(self):
        html = """
        <form class="pagination__jumpwrap">
            <input name="page" max="15">
        </form>
        """
        assert _parse_max_page(html) == 15

    def test_no_pagination(self):
        assert _parse_max_page("<html></html>") == 1

    def test_invalid_max_value(self):
        html = """
        <form class="pagination__jumpwrap">
            <input name="page" max="abc">
        </form>
        """
        assert _parse_max_page(html) == 1

    def test_missing_max_attribute(self):
        html = """
        <form class="pagination__jumpwrap">
            <input name="page">
        </form>
        """
        assert _parse_max_page(html) == 1
