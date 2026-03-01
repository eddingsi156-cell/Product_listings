"""产品拆分 — 聚类逻辑 + 文件移动"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)

from ..config import CLUSTER_THRESHOLD_DEFAULT, CLUSTER_THRESHOLD_MAX, CLUSTER_THRESHOLD_MIN, COMBINED_DIM, IMAGE_EXTS
from ..image_processor import list_images  # 统一定义在 image_processor 中
from ..organizer import sanitize_folder_name, unique_folder_path
from .feature_extractor import get_extractor


@dataclass
class BatchScanItem:
    """批量扫描单个文件夹的结果。"""
    folder: Path
    image_count: int
    group_count: int
    result: SplitResult
    error: str | None  # None = 成功


@dataclass
class SplitGroup:
    """单个拆分分组"""
    id: int
    name: str
    image_paths: list[Path]
    original_indices: list[int]


@dataclass
class SplitResult:
    """拆分结果（含缓存特征，支持快速重新聚类）"""
    album_folder: Path
    groups: list[SplitGroup]
    image_paths: list[Path]
    features: np.ndarray  # (N, 608) 缓存


def cluster_images(features: np.ndarray, threshold: float) -> np.ndarray:
    """余弦距离 + 层次聚类，返回标签数组。

    Args:
        features: (N, D) L2 归一化特征。
        threshold: 距离阈值，越小分组越多。

    Returns:
        (N,) int32 标签。
    """
    n = features.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.int32)

    if n > 2000:
        logger.warning(
            "聚类输入过大（%d 张），N×N 距离矩阵将占用 ~%d MB 内存",
            n, n * n * 4 // (1024 * 1024),
        )

    # 余弦距离 = 1 - 余弦相似度（特征已 L2 归一化，dot = cosine sim）
    cosine_sim = features @ features.T
    cosine_dist = 1.0 - cosine_sim
    np.fill_diagonal(cosine_dist, 0.0)
    cosine_dist = np.clip(cosine_dist, 0.0, 2.0)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(cosine_dist)
    return labels.astype(np.int32)


def build_split_result(
    folder: Path,
    paths: list[Path],
    features: np.ndarray,
    labels: np.ndarray,
) -> SplitResult:
    """根据聚类标签构建 SplitResult。"""
    groups: list[SplitGroup] = []
    unique_labels = sorted(set(labels.tolist()))

    for group_id, label in enumerate(unique_labels):
        indices = [i for i, lb in enumerate(labels) if lb == label]
        group = SplitGroup(
            id=group_id,
            name=f"配色_{group_id + 1}",
            image_paths=[paths[i] for i in indices],
            original_indices=indices,
        )
        groups.append(group)

    return SplitResult(
        album_folder=folder,
        groups=groups,
        image_paths=paths,
        features=features,
    )


def recluster(result: SplitResult, threshold: float) -> SplitResult:
    """用缓存特征重新聚类（主线程同步调用，<10ms）。"""
    threshold = max(CLUSTER_THRESHOLD_MIN, min(CLUSTER_THRESHOLD_MAX, threshold))
    labels = cluster_images(result.features, threshold)
    return build_split_result(
        result.album_folder, result.image_paths, result.features, labels,
    )


def apply_split(
    result: SplitResult,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """将拆分结果应用到文件系统，移动文件到平级文件夹。

    拆分后的文件夹创建在原产品文件夹的父目录下（平级），
    命名格式为 "原文件夹名 分组名"。原产品文件夹清空后删除。

    Returns:
        创建的文件夹路径列表。
    """
    album_folder = result.album_folder
    parent = album_folder.parent  # downloads 目录
    album_name = album_folder.name
    created_folders: list[Path] = []
    total = sum(len(g.image_paths) for g in result.groups)
    done = 0

    for group in result.groups:
        folder_name = f"{album_name} {group.name}"
        folder_path, _ = unique_folder_path(parent, folder_name)
        folder_path.mkdir(parents=True, exist_ok=True)
        created_folders.append(folder_path)

        for img_path in group.image_paths:
            dest = folder_path / img_path.name
            if img_path.exists():
                shutil.move(str(img_path), str(dest))
            done += 1
            if on_progress:
                on_progress(done, total)

    # 原产品文件夹已清空，删除
    try:
        if album_folder.exists() and not any(album_folder.iterdir()):
            album_folder.rmdir()
    except OSError:
        pass  # 目录非空或权限不足，忽略

    return created_folders


def extract_and_split(
    folder: Path,
    threshold: float = CLUSTER_THRESHOLD_DEFAULT,
    on_status: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> SplitResult:
    """完整流水线：列出图片 → 加载模型 → 提取特征 → 聚类。

    Args:
        folder: 相册文件夹路径。
        threshold: 聚类距离阈值。
        on_status: 状态文字回调。
        on_progress: 特征提取进度回调 (当前, 总数)。
    """
    images = list_images(folder)
    if not images:
        return SplitResult(
            album_folder=folder, groups=[], image_paths=[], features=np.empty((0, COMBINED_DIM)),
        )

    extractor = get_extractor()

    if on_status:
        on_status("正在加载模型...")
    extractor.load_model(on_progress=on_status)

    if on_status:
        on_status("正在提取特征...")
    features = extractor.extract_combined_batch(images, on_progress=on_progress)

    if on_status:
        on_status("正在聚类...")
    labels = cluster_images(features, threshold)

    result = build_split_result(folder, images, features, labels)

    if on_status:
        on_status(f"拆分完成：{len(result.groups)} 个分组")

    return result


def batch_extract_and_split(
    folders: list[Path],
    threshold: float,
    is_cancelled: Callable[[], bool] | None = None,
    on_folder_start: Callable[[int, int, Path], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    on_image_progress: Callable[[int, int], None] | None = None,
    on_folder_done: Callable[[int, int, BatchScanItem], None] | None = None,
) -> list[BatchScanItem]:
    """批量扫描多个文件夹：加载模型一次 → 逐文件夹提取特征+聚类。

    Args:
        folders: 待扫描的文件夹列表。
        threshold: 聚类距离阈值。
        is_cancelled: 取消检查回调，返回 True 时中止。
        on_folder_start: 开始处理某文件夹时的回调 (当前索引, 总数, 路径)。
        on_status: 状态文字回调。
        on_image_progress: 图片级进度回调 (当前, 总数)。
        on_folder_done: 单个文件夹处理完成回调 (当前索引, 总数, 结果)。

    Returns:
        所有文件夹的扫描结果列表。
    """
    results: list[BatchScanItem] = []
    total = len(folders)

    if total == 0:
        return results

    # 加载模型一次
    extractor = get_extractor()
    if on_status:
        on_status("正在加载模型...")
    extractor.load_model(on_progress=on_status)

    for idx, folder in enumerate(folders):
        if is_cancelled and is_cancelled():
            break

        if on_folder_start:
            on_folder_start(idx, total, folder)

        try:
            images = list_images(folder)
            if not images:
                result = SplitResult(
                    album_folder=folder, groups=[], image_paths=[],
                    features=np.empty((0, COMBINED_DIM)),
                )
                item = BatchScanItem(
                    folder=folder, image_count=0, group_count=0,
                    result=result, error=None,
                )
            else:
                if on_status:
                    on_status(f"提取特征: {folder.name}")
                features = extractor.extract_combined_batch(
                    images, on_progress=on_image_progress,
                )

                if is_cancelled and is_cancelled():
                    break

                labels = cluster_images(features, threshold)
                result = build_split_result(folder, images, features, labels)
                item = BatchScanItem(
                    folder=folder,
                    image_count=len(images),
                    group_count=len(result.groups),
                    result=result,
                    error=None,
                )
        except Exception as e:
            # 单个文件夹出错不影响整体
            empty_result = SplitResult(
                album_folder=folder, groups=[], image_paths=[],
                features=np.empty((0, COMBINED_DIM)),
            )
            item = BatchScanItem(
                folder=folder, image_count=0, group_count=0,
                result=empty_result, error=str(e),
            )

        results.append(item)
        if on_folder_done:
            on_folder_done(idx, total, item)

    return results
