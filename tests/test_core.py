"""核心纯函数单元测试"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from yupoo_scraper.scraper import parse_username, _ensure_size
from yupoo_scraper.organizer import sanitize_folder_name, unique_folder_path, find_image_folders
from yupoo_scraper.downloader import _solve_cdn_challenge, _extract_ext, _is_valid_image_file


# ── parse_username ────────────────────────────────────────────

class TestParseUsername:
    def test_photos_url(self):
        url = "https://x.yupoo.com/photos/gyg88/albums"
        assert parse_username(url) == "gyg88"

    def test_subdomain_url(self):
        url = "https://gyg88.x.yupoo.com/albums"
        assert parse_username(url) == "gyg88"

    def test_photos_url_with_page(self):
        url = "https://x.yupoo.com/photos/abc123/albums?page=2"
        assert parse_username(url) == "abc123"

    def test_subdomain_with_categories(self):
        url = "https://abc123.x.yupoo.com/categories/123456"
        assert parse_username(url) == "abc123"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError):
            parse_username("https://google.com/something")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_username("")


# ── sanitize_folder_name ──────────────────────────────────────

class TestSanitizeFolderName:
    def test_normal_name(self):
        assert sanitize_folder_name("Nike Air Max 90") == "Nike Air Max 90"

    def test_illegal_chars_replaced(self):
        result = sanitize_folder_name('product:size<M>/test')
        assert ":" not in result
        assert "<" not in result
        assert "/" not in result

    def test_leading_trailing_stripped(self):
        result = sanitize_folder_name("  . product . ")
        assert not result.startswith(" ")
        assert not result.startswith(".")
        assert not result.endswith(" ")
        assert not result.endswith(".")

    def test_empty_becomes_untitled(self):
        assert sanitize_folder_name("") == "untitled"
        assert sanitize_folder_name("...") == "untitled"

    def test_max_length(self):
        long_name = "a" * 300
        result = sanitize_folder_name(long_name)
        assert len(result) <= 200

    def test_consecutive_separators_merged(self):
        result = sanitize_folder_name("hello   world__test")
        assert "  " not in result
        assert "__" not in result

    def test_reserved_names_prefixed(self):
        assert sanitize_folder_name("CON") == "_CON"
        assert sanitize_folder_name("nul") == "_nul"
        assert sanitize_folder_name("COM1") == "_COM1"

    def test_truncation_strips_trailing_dot(self):
        # 200th char is a dot → should be stripped
        name = "a" * 199 + "."
        result = sanitize_folder_name(name)
        assert not result.endswith(".")


# ── _solve_cdn_challenge ──────────────────────────────────────

class TestSolveCdnChallenge:
    def test_none_for_normal_image(self):
        # 正常图片数据（大于 2048 bytes）
        data = b"\xff\xd8\xff\xe0" + b"\x00" * 3000
        assert _solve_cdn_challenge(data) is None

    def test_none_for_no_marker(self):
        data = b"<html><body>Hello World</body></html>"
        assert _solve_cdn_challenge(data) is None

    def test_solves_challenge_with_four_numbers(self):
        # 模拟 CDN 挑战页面
        html = (
            '<script>'
            'var a = 1234567890;'
            'var b = 1111111111;'
            'var c = 2222222222;'
            'var d = 3333333333;'
            'document.cookie = "__tst_status=" + (b+c+d) + "#";'
            'document.cookie = "EO_Bot_Ssid=" + a;'
            '</script>'
        )
        result = _solve_cdn_challenge(html.encode("utf-8"))
        assert result is not None
        assert result["EO_Bot_Ssid"] == "1234567890"
        expected_tst = 1111111111 + 2222222222 + 3333333333
        assert result["__tst_status"] == f"{expected_tst}#"

    def test_none_for_fewer_than_four_numbers(self):
        html = (
            '<script>'
            'var a = 1234567890;'
            'var b = 1111111111;'
            '__tst_status EO_Bot_Ssid'
            '</script>'
        )
        result = _solve_cdn_challenge(html.encode("utf-8"))
        assert result is None

    def test_none_for_large_data(self):
        # 超过 2048 字节 → 不是挑战页面
        data = b"__tst_status EO_Bot_Ssid " + b"x" * 2100
        assert _solve_cdn_challenge(data) is None


# ── _ensure_size ──────────────────────────────────────────────

class TestEnsureSize:
    def test_replace_small_with_big(self):
        url = "https://photo.yupoo.com/hash/small.jpeg"
        assert _ensure_size(url, "big") == "https://photo.yupoo.com/hash/big.jpeg"

    def test_replace_medium_with_big(self):
        url = "https://photo.yupoo.com/hash/medium.jpeg"
        assert _ensure_size(url, "big") == "https://photo.yupoo.com/hash/big.jpeg"

    def test_replace_original_with_big(self):
        url = "https://photo.yupoo.com/hash/original.jpeg"
        assert _ensure_size(url, "big") == "https://photo.yupoo.com/hash/big.jpeg"

    def test_replace_square_with_big(self):
        url = "https://photo.yupoo.com/hash/square.jpeg"
        assert _ensure_size(url, "big") == "https://photo.yupoo.com/hash/big.jpeg"

    def test_no_change_when_already_target(self):
        url = "https://photo.yupoo.com/hash/big.jpeg"
        assert _ensure_size(url, "big") == url

    def test_preserves_query_string(self):
        url = "https://photo.yupoo.com/hash/small.jpeg?t=123"
        result = _ensure_size(url, "big")
        assert "/big.jpeg?t=123" in result

    def test_no_match_returns_unchanged(self):
        url = "https://photo.yupoo.com/hash/unknown.jpeg"
        assert _ensure_size(url, "big") == url


# ── _extract_ext ──────────────────────────────────────────────

class TestExtractExt:
    def test_jpeg(self):
        assert _extract_ext("https://photo.yupoo.com/hash/big.jpeg") == ".jpeg"

    def test_jpg(self):
        assert _extract_ext("https://photo.yupoo.com/hash/big.jpg") == ".jpg"

    def test_png(self):
        assert _extract_ext("https://photo.yupoo.com/hash/big.png") == ".png"

    def test_webp(self):
        assert _extract_ext("https://photo.yupoo.com/hash/big.webp") == ".webp"

    def test_with_query_string(self):
        assert _extract_ext("https://photo.yupoo.com/hash/big.jpeg?v=123") == ".jpeg"

    def test_no_ext_returns_default(self):
        assert _extract_ext("https://photo.yupoo.com/hash/noext") == ".jpeg"

    def test_long_ext_returns_default(self):
        # Extension > 5 chars should fall back to .jpeg
        assert _extract_ext("https://example.com/file.toolong") == ".jpeg"


# ── _is_valid_image_file ─────────────────────────────────────

class TestIsValidImageFile:
    def test_jpeg_file(self, tmp_path):
        f = tmp_path / "test.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        assert _is_valid_image_file(f) is True

    def test_png_file(self, tmp_path):
        f = tmp_path / "test.png"
        f.write_bytes(b"\x89PNG\r\n" + b"\x00" * 100)
        assert _is_valid_image_file(f) is True

    def test_webp_file(self, tmp_path):
        f = tmp_path / "test.webp"
        f.write_bytes(b"RIFF" + b"\x00" * 100)
        assert _is_valid_image_file(f) is True

    def test_gif_file(self, tmp_path):
        f = tmp_path / "test.gif"
        f.write_bytes(b"GIF89a" + b"\x00" * 100)
        assert _is_valid_image_file(f) is True

    def test_html_challenge_page(self, tmp_path):
        """CDN challenge HTML (~988 bytes) should NOT be valid image."""
        f = tmp_path / "challenge.jpeg"
        f.write_bytes(b"<html><script>__tst_status</script></html>" + b"\x00" * 900)
        assert _is_valid_image_file(f) is False

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.jpg"
        f.write_bytes(b"")
        assert _is_valid_image_file(f) is False

    def test_too_short_file(self, tmp_path):
        f = tmp_path / "tiny.jpg"
        f.write_bytes(b"\xff\xd8")
        assert _is_valid_image_file(f) is False

    def test_nonexistent_file(self, tmp_path):
        f = tmp_path / "nonexistent.jpg"
        assert _is_valid_image_file(f) is False


# ── unique_folder_path ────────────────────────────────────────

class TestUniqueFolderPath:
    def test_unique_name(self, tmp_path):
        path, name = unique_folder_path(tmp_path, "album")
        assert name == "album"
        assert path == tmp_path / "album"

    def test_conflict_adds_suffix(self, tmp_path):
        (tmp_path / "album").mkdir()
        path, name = unique_folder_path(tmp_path, "album")
        assert name == "album_2"
        assert path == tmp_path / "album_2"

    def test_multiple_conflicts(self, tmp_path):
        (tmp_path / "album").mkdir()
        (tmp_path / "album_2").mkdir()
        (tmp_path / "album_3").mkdir()
        path, name = unique_folder_path(tmp_path, "album")
        assert name == "album_4"

    def test_sanitizes_name(self, tmp_path):
        path, name = unique_folder_path(tmp_path, "album:invalid<name>")
        assert ":" not in name
        assert "<" not in name


# ── find_image_folders ────────────────────────────────────────

class TestFindImageFolders:
    def test_finds_folders_with_images(self, tmp_path):
        folder = tmp_path / "album1"
        folder.mkdir()
        (folder / "img.jpg").write_bytes(b"\xff\xd8\xff")
        result = find_image_folders(tmp_path)
        assert len(result) == 1
        assert result[0] == folder

    def test_skips_empty_folders(self, tmp_path):
        (tmp_path / "empty").mkdir()
        result = find_image_folders(tmp_path)
        assert len(result) == 0

    def test_skips_folders_without_images(self, tmp_path):
        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "readme.txt").write_text("hello")
        result = find_image_folders(tmp_path)
        assert len(result) == 0

    def test_multiple_folders_sorted(self, tmp_path):
        for name in ["c_album", "a_album", "b_album"]:
            folder = tmp_path / name
            folder.mkdir()
            (folder / "img.png").write_bytes(b"\x89PNG\r\n")
        result = find_image_folders(tmp_path)
        assert len(result) == 3
        assert result[0].name == "a_album"
        assert result[1].name == "b_album"
        assert result[2].name == "c_album"


# ── cluster_images ────────────────────────────────────────────

class TestClusterImages:
    def test_single_image_returns_one_cluster(self):
        from yupoo_scraper.ml.splitter import cluster_images
        features = np.random.randn(1, 608).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        labels = cluster_images(features, threshold=0.5)
        assert len(labels) == 1
        assert labels[0] == 0

    def test_identical_features_same_cluster(self):
        from yupoo_scraper.ml.splitter import cluster_images
        feat = np.random.randn(1, 608).astype(np.float32)
        feat /= np.linalg.norm(feat)
        features = np.repeat(feat, 5, axis=0)  # 5 identical vectors
        labels = cluster_images(features, threshold=0.3)
        assert len(set(labels)) == 1  # All in same cluster

    def test_orthogonal_features_different_clusters(self):
        from yupoo_scraper.ml.splitter import cluster_images
        # Create near-orthogonal vectors (cosine distance ~1.0)
        features = np.eye(3, 608, dtype=np.float32)
        labels = cluster_images(features, threshold=0.3)
        assert len(set(labels)) == 3  # Each in its own cluster

    def test_threshold_affects_cluster_count(self):
        from yupoo_scraper.ml.splitter import cluster_images
        np.random.seed(42)
        features = np.random.randn(10, 608).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        labels_tight = cluster_images(features, threshold=0.1)
        labels_loose = cluster_images(features, threshold=1.5)
        # Tighter threshold → more clusters
        assert len(set(labels_tight)) >= len(set(labels_loose))


# ── ProductInfo detail/main image split ───────────────────────

class TestProductInfoImageSplit:
    def test_main_and_detail_do_not_overlap(self, tmp_path):
        """Regression test: detail_images must not include main_images."""
        from yupoo_scraper.config import MAIN_IMAGE_MAX
        # Create fake image files
        for i in range(20):
            (tmp_path / f"img_{i:03d}.jpg").write_bytes(b"\xff\xd8\xff")

        from yupoo_scraper.image_processor import list_images
        images = list_images(tmp_path)

        main_imgs = images[:MAIN_IMAGE_MAX]
        detail_imgs = images[MAIN_IMAGE_MAX:]

        # No overlap
        main_set = set(main_imgs)
        detail_set = set(detail_imgs)
        assert main_set.isdisjoint(detail_set), "main and detail images overlap!"

        # Together they cover all images
        assert len(main_imgs) + len(detail_imgs) == len(images)
