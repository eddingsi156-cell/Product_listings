"""image_processor 模块单元测试"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from yupoo_scraper.image_processor import (
    list_images,
    pad_to_square,
    save_image,
    is_white_background,
    reorder_main_image,
)


# ── list_images ──────────────────────────────────────────────


class TestListImages:
    def test_returns_sorted_images(self, tmp_path):
        for name in ["c.jpg", "a.png", "b.jpeg"]:
            (tmp_path / name).write_bytes(b"\xff\xd8\xff")
        result = list_images(tmp_path)
        assert [p.name for p in result] == ["a.png", "b.jpeg", "c.jpg"]

    def test_ignores_non_image_files(self, tmp_path):
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("a,b")
        (tmp_path / "img.jpg").write_bytes(b"\xff\xd8\xff")
        result = list_images(tmp_path)
        assert len(result) == 1
        assert result[0].name == "img.jpg"

    def test_empty_folder(self, tmp_path):
        assert list_images(tmp_path) == []

    def test_recognizes_all_extensions(self, tmp_path):
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            (tmp_path / f"img{ext}").write_bytes(b"\x00" * 10)
        result = list_images(tmp_path)
        assert len(result) == 5


# ── pad_to_square ────────────────────────────────────────────


class TestPadToSquare:
    def test_landscape_image_padded(self, tmp_path):
        img = Image.new("RGB", (200, 100), (255, 0, 0))
        path = tmp_path / "landscape.jpg"
        img.save(path)
        assert pad_to_square(path) is True
        result = Image.open(path)
        assert result.size == (200, 200)
        result.close()

    def test_portrait_image_padded(self, tmp_path):
        img = Image.new("RGB", (100, 200), (0, 0, 255))
        path = tmp_path / "portrait.jpg"
        img.save(path)
        assert pad_to_square(path) is True
        result = Image.open(path)
        assert result.size == (200, 200)
        result.close()

    def test_square_image_skipped(self, tmp_path):
        img = Image.new("RGB", (100, 100), (0, 255, 0))
        path = tmp_path / "square.jpg"
        img.save(path)
        assert pad_to_square(path) is False

    def test_rgba_image_converted(self, tmp_path):
        img = Image.new("RGBA", (200, 100), (255, 0, 0, 128))
        path = tmp_path / "rgba.png"
        img.save(path)
        assert pad_to_square(path) is True
        result = Image.open(path)
        assert result.size == (200, 200)
        assert result.mode == "RGB"
        result.close()

    def test_fill_color_applied(self, tmp_path):
        img = Image.new("RGB", (100, 50), (255, 0, 0))
        path = tmp_path / "fill.jpg"
        img.save(path)
        pad_to_square(path, fill_color=(0, 0, 0))
        result = Image.open(path)
        # 检查角落像素（应该是填充色黑色）
        pixel = result.getpixel((0, 0))
        assert pixel[0] < 10  # 接近黑色（JPEG 有压缩误差）
        result.close()


# ── save_image ───────────────────────────────────────────────


class TestSaveImage:
    def test_saves_jpeg(self, tmp_path):
        img = Image.new("RGB", (50, 50), (128, 128, 128))
        path = tmp_path / "test.jpg"
        save_image(img, path, quality=90)
        assert path.exists()
        loaded = Image.open(path)
        assert loaded.size == (50, 50)
        loaded.close()

    def test_saves_png(self, tmp_path):
        img = Image.new("RGB", (50, 50), (128, 128, 128))
        path = tmp_path / "test.png"
        save_image(img, path, quality=90)
        assert path.exists()

    def test_atomic_write_no_leftover_tmp(self, tmp_path):
        img = Image.new("RGB", (50, 50))
        path = tmp_path / "atomic.jpg"
        save_image(img, path, quality=90)
        # 临时文件不应留下
        tmp_file = path.with_suffix(".jpg.tmp")
        assert not tmp_file.exists()

    def test_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "overwrite.jpg"
        # 先写一个小文件
        img1 = Image.new("RGB", (10, 10))
        save_image(img1, path, quality=90)
        size1 = path.stat().st_size
        # 再写一个大文件
        img2 = Image.new("RGB", (500, 500))
        save_image(img2, path, quality=90)
        size2 = path.stat().st_size
        assert size2 > size1


# ── is_white_background ──────────────────────────────────────


class TestIsWhiteBackground:
    def test_white_image_detected(self, tmp_path):
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        path = tmp_path / "white.jpg"
        img.save(path)
        assert is_white_background(path)

    def test_colored_image_not_detected(self, tmp_path):
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        path = tmp_path / "red.jpg"
        img.save(path)
        assert not is_white_background(path)

    def test_white_border_with_center_content(self, tmp_path):
        """白色边框 + 彩色中心 → 应检测为白底"""
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        # 在中心 50% 区域画彩色
        for x in range(50, 150):
            for y in range(50, 150):
                img.putpixel((x, y), (255, 0, 0))
        path = tmp_path / "white_border.jpg"
        img.save(path)
        assert is_white_background(path)


# ── reorder_main_image ───────────────────────────────────────


class TestReorderMainImage:
    def test_selected_image_becomes_first(self, tmp_path):
        for i in range(1, 4):
            img = Image.new("RGB", (10, 10))
            img.save(tmp_path / f"img_{i}.jpg")
        target = tmp_path / "img_3.jpg"
        reorder_main_image(tmp_path, target)
        result = list_images(tmp_path)
        # 第一个文件的文件名应包含 img_3
        assert "img_3" in result[0].name

    def test_empty_folder_no_error(self, tmp_path):
        """空文件夹应该静默返回"""
        reorder_main_image(tmp_path, tmp_path / "nonexistent.jpg")

    def test_all_files_preserved(self, tmp_path):
        names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
        for name in names:
            img = Image.new("RGB", (10, 10))
            img.save(tmp_path / name)
        reorder_main_image(tmp_path, tmp_path / "c.jpg")
        result = list_images(tmp_path)
        assert len(result) == 4

    def test_files_numbered_sequentially(self, tmp_path):
        for name in ["x.jpg", "y.jpg", "z.jpg"]:
            img = Image.new("RGB", (10, 10))
            img.save(tmp_path / name)
        reorder_main_image(tmp_path, tmp_path / "y.jpg")
        result = list_images(tmp_path)
        # 文件应以 001_, 002_, 003_ 开头
        assert result[0].name.startswith("001_")
        assert result[1].name.startswith("002_")
        assert result[2].name.startswith("003_")
