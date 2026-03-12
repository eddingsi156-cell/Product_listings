"""图片处理 — 1:1 正方形补齐（白色填充，原地覆盖）+ 白底检测 / 去背景"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from . import config
from .config import IMAGE_EXTS

logger = logging.getLogger(__name__)



@dataclass
class FolderProcessResult:
    """单个文件夹的处理结果"""

    folder: Path
    total: int = 0
    processed: int = 0
    skipped: int = 0       # 已是正方形，无需处理
    failed: int = 0
    cancelled: bool = False
    errors: list[str] = field(default_factory=list)


def list_images(folder: Path) -> list[Path]:
    """列出文件夹中所有图片文件，按文件名排序。"""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def pad_to_square(
    path: Path,
    fill_color: tuple[int, int, int] = config.SQUARE_FILL_COLOR,
    quality: int = config.SQUARE_JPEG_QUALITY,
) -> bool:
    """将单张图片补齐为 1:1 正方形，原地覆盖。

    Returns:
        True 表示实际做了补齐处理，False 表示已是正方形（跳过）
    """
    with Image.open(path) as raw_img:
        raw_img.load()  # 强制读入内存，立即释放文件句柄

        # 保留 RGBA/LA/P 模式的透明度信息，统一转为 RGB 输出
        if raw_img.mode in ("RGBA", "LA", "P"):
            img = Image.new("RGB", raw_img.size, fill_color)
            conv = raw_img.convert("RGBA") if raw_img.mode != "RGBA" else raw_img
            img.paste(conv, mask=conv.split()[3])
        elif raw_img.mode != "RGB":
            img = raw_img.convert("RGB")
        else:
            img = raw_img.copy()

    w, h = img.size

    if w == h:
        img.close()
        return False  # 已是正方形，跳过

    size = max(w, h)
    canvas = Image.new("RGB", (size, size), fill_color)
    offset = ((size - w) // 2, (size - h) // 2)
    canvas.paste(img, offset)
    img.close()
    save_image(canvas, path, quality)
    canvas.close()
    return True


def save_image(img: Image.Image, dst: Path, quality: int) -> None:
    """根据后缀选择合适格式保存（先写临时文件，再原子替换）。"""
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    try:
        ext = dst.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            img.save(tmp_path, format="JPEG", quality=quality)
        elif ext == ".png":
            img.save(tmp_path, format="PNG")
        elif ext == ".webp":
            img.save(tmp_path, format="WEBP", quality=quality)
        elif ext == ".bmp":
            img.save(tmp_path, format="BMP")
        else:
            img.save(tmp_path, format="JPEG", quality=quality)
        os.replace(str(tmp_path), str(dst))
    except BaseException:
        # 保存失败时清理临时文件，不损坏原文件
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def process_folder(
    folder: Path,
    fill_color: tuple[int, int, int] = config.SQUARE_FILL_COLOR,
    quality: int = config.SQUARE_JPEG_QUALITY,
    on_image_done: Callable[[int, int, Path], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> FolderProcessResult:
    """处理单个文件夹：所有图片补齐为 1:1 正方形（原地覆盖）。"""
    images = list_images(folder)
    result = FolderProcessResult(folder=folder, total=len(images))

    if not images:
        return result

    for i, img_path in enumerate(images):
        if is_cancelled and is_cancelled():
            result.cancelled = True
            result.total = i  # 更新为实际已遍历数
            break

        try:
            padded = pad_to_square(img_path, fill_color, quality)
            if padded:
                result.processed += 1
            else:
                result.skipped += 1
        except Exception as e:
            result.failed += 1
            result.errors.append(f"{img_path.name}: {e}")

        if on_image_done:
            on_image_done(i + 1, len(images), img_path)

    return result


def batch_process(
    folders: list[Path],
    fill_color: tuple[int, int, int] = config.SQUARE_FILL_COLOR,
    quality: int = config.SQUARE_JPEG_QUALITY,
    is_cancelled: Callable[[], bool] | None = None,
    on_folder_start: Callable[[int, int, Path], None] | None = None,
    on_image_done: Callable[[int, int, Path], None] | None = None,
    on_folder_done: Callable[[int, int, FolderProcessResult], None] | None = None,
) -> list[FolderProcessResult]:
    """批量处理多个文件夹（原地覆盖）。"""
    results: list[FolderProcessResult] = []

    for idx, folder in enumerate(folders):
        if is_cancelled and is_cancelled():
            break

        if on_folder_start:
            on_folder_start(idx, len(folders), folder)

        result = process_folder(
            folder,
            fill_color=fill_color,
            quality=quality,
            on_image_done=on_image_done,
            is_cancelled=is_cancelled,
        )
        results.append(result)

        if on_folder_done:
            on_folder_done(idx, len(folders), result)

    return results


# ── 白底检测 / 去背景 / 主图重排序 ─────────────────────────────


def is_white_background(
    path: Path,
    threshold: int = config.WHITE_BG_THRESHOLD,
    edge_ratio: float = config.WHITE_BG_EDGE_RATIO,
) -> bool:
    """检测图片是否为白底：取边缘 5% 像素，判断白色像素占比。"""
    with Image.open(path) as raw_img:
        img = raw_img.convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    margin_y = max(1, int(h * 0.05))
    margin_x = max(1, int(w * 0.05))

    # 取四条边缘区域
    top = arr[:margin_y, :, :]
    bottom = arr[h - margin_y:, :, :]
    left = arr[margin_y:h - margin_y, :margin_x, :]
    right = arr[margin_y:h - margin_y, w - margin_x:, :]

    edge_pixels = np.concatenate(
        [top.reshape(-1, 3), bottom.reshape(-1, 3),
         left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0,
    )
    white_mask = np.all(edge_pixels >= threshold, axis=1)
    ratio = white_mask.sum() / len(white_mask)
    return ratio >= edge_ratio


_rembg_sessions: dict[str, object] = {}
_rembg_lock = threading.Lock()


def remove_background_to_white(
    path: Path,
    fill_color: tuple[int, int, int] = (255, 255, 255),
    model_name: str = "isnet-general-use",
) -> Image.Image:
    """使用 rembg 去除背景并填充白色，返回 PIL Image（不写盘）。

    Args:
        model_name: rembg 模型，推荐 "isnet-general-use"（效果好、速度快）。
    """
    from rembg import new_session, remove

    img = Image.open(path).convert("RGB")
    with _rembg_lock:
        if model_name not in _rembg_sessions:
            _rembg_sessions[model_name] = new_session(model_name)
        session = _rembg_sessions[model_name]
    # rembg 返回 RGBA；开启 alpha_matting 细化边缘
    result = remove(
        img,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=20,
        alpha_matting_erode_size=10,
    )
    # 合成到白底
    bg = Image.new("RGB", result.size, fill_color)
    bg.paste(result, mask=result.split()[3])
    return bg


def reorder_main_image(folder: Path, main_image: Path) -> None:
    """重排序：将 main_image 排到第一位（001_ 前缀）。

    两阶段重命名防止冲突：先改为临时名，再改为最终名。
    """
    images = list_images(folder)
    if not images:
        return

    # 去掉已有的数字前缀，获得干净文件名
    def clean_name(p: Path) -> str:
        return re.sub(r"^\d{3}_", "", p.name)

    # 建立排序列表：main_image 在最前
    ordered: list[Path] = []
    rest: list[Path] = []
    for img in images:
        if img.resolve() == main_image.resolve():
            ordered.insert(0, img)
        else:
            rest.append(img)
    ordered.extend(rest)

    # 阶段 1：全部改为临时名（防止重名冲突）
    tmp_pairs: list[tuple[Path, str]] = []
    renamed_originals: list[tuple[Path, Path]] = []  # (tmp_path, original_path) 用于回滚
    try:
        for i, img in enumerate(ordered):
            tmp_name = f"_tmp_{i:03d}_{clean_name(img)}"
            tmp_path = img.parent / tmp_name
            img.rename(tmp_path)
            renamed_originals.append((tmp_path, img))
            tmp_pairs.append((tmp_path, clean_name(img)))
    except OSError:
        # 阶段 1 中途失败，回滚已改名的文件
        for tmp_path, original_path in reversed(renamed_originals):
            try:
                tmp_path.rename(original_path)
            except OSError:
                pass
        raise

    # 阶段 2：改为最终名（带序号前缀）
    renamed_finals: list[tuple[Path, Path]] = []  # (final_path, tmp_path) 用于回滚
    try:
        for i, (tmp_path, base_name) in enumerate(tmp_pairs):
            final_name = f"{i + 1:03d}_{base_name}"
            final_path = tmp_path.parent / final_name
            tmp_path.rename(final_path)
            renamed_finals.append((final_path, tmp_path))
    except OSError:
        # 阶段 2 失败 → 回滚到原始文件名
        for final_path, tmp_path_rb in reversed(renamed_finals):
            try:
                final_path.rename(tmp_path_rb)
            except OSError:
                pass
        for tmp_path_rb, original_path in reversed(renamed_originals):
            try:
                tmp_path_rb.rename(original_path)
            except OSError:
                pass
        raise
