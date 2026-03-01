"""文件整理 — 文件夹命名、非法字符清理、文件夹扫描"""

from __future__ import annotations

import re
from pathlib import Path

from .config import IMAGE_EXTS


# Windows 文件名非法字符
_ILLEGAL_CHARS = re.compile(r'[\\/:*?"<>|]')
# 连续空格/下划线
_MULTI_SEP = re.compile(r"[_ ]{2,}")
# Windows 保留文件名（不区分大小写）
_RESERVED_NAMES = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})


def sanitize_folder_name(name: str) -> str:
    """清理产品名称，使其可作为文件夹名。

    - 替换 Windows 非法字符为下划线
    - 去除首尾空白和句点
    - 合并连续分隔符
    - 处理 Windows 保留名（CON, PRN, AUX, NUL, COM1-9, LPT1-9）
    - 截断至 200 字符（NTFS 限制）
    """
    name = name.strip()
    name = _ILLEGAL_CHARS.sub("_", name)
    name = _MULTI_SEP.sub("_", name)
    name = name.strip(" _.")

    if not name:
        name = "untitled"

    # Windows 保留名检测（CON, CON.txt 等均不允许）
    stem = name.split(".")[0].upper()
    if stem in _RESERVED_NAMES:
        name = f"_{name}"

    # NTFS 允许 255 字符，留点余量；截断后再清理尾部句点
    name = name[:200].rstrip(" _.")
    return name or "untitled"


def unique_folder_path(base_dir: Path, name: str) -> tuple[Path, str]:
    """确保文件夹不重名，必要时追加 _2, _3 ...

    Returns:
        (完整路径, 最终文件夹名)
    """
    folder_name = sanitize_folder_name(name)
    path = base_dir / folder_name

    if not path.exists():
        return path, folder_name

    for counter in range(2, 10002):
        new_name = f"{folder_name}_{counter}"
        new_path = base_dir / new_name
        if not new_path.exists():
            return new_path, new_name

    raise RuntimeError(f"无法为 '{folder_name}' 生成唯一文件夹名（尝试超过 10000 次）")


def find_image_folders(base_dir: Path) -> list[Path]:
    """扫描一级子文件夹，返回含图片的文件夹路径列表。"""
    results = []
    for sub in sorted(base_dir.iterdir()):
        if not sub.is_dir():
            continue
        has_images = any(
            f.is_file() and f.suffix.lower() in IMAGE_EXTS
            for f in sub.iterdir()
        )
        if has_images:
            results.append(sub)
    return results
