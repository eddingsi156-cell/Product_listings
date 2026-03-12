"""产品拆分 — 聚类逻辑 + 文件移动"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from datetime import datetime

import numpy as np
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)

from ..config import CLUSTER_THRESHOLD_DEFAULT, CLUSTER_THRESHOLD_MAX, CLUSTER_THRESHOLD_MIN, COMBINED_DIM, IMAGE_EXTS
from ..image_processor import list_images  # 统一定义在 image_processor 中
from ..organizer import sanitize_folder_name, unique_folder_path
from .feature_extractor import get_extractor
from .split_history import get_split_history


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


def _cosine_distance_matrix(features: np.ndarray) -> np.ndarray:
    """计算余弦距离矩阵（向量化矩阵乘法）。

    Args:
        features: (N, D) L2 归一化特征。

    Returns:
        (N, N) float32 余弦距离矩阵。
    """
    cosine_sim = features.astype(np.float32) @ features.astype(np.float32).T
    cosine_dist = 1.0 - cosine_sim
    np.fill_diagonal(cosine_dist, 0.0)
    cosine_dist = np.clip(cosine_dist, 0.0, 2.0)
    return cosine_dist


# 层次聚类的最大样本数上限 — 超过此值用 KMeans 拆分后再层次聚类
# 200 样本的距离矩阵 = 200×200×4 = 160KB，计算快且内存安全
_HIERARCHICAL_MAX = 200


def cluster_images(features: np.ndarray, threshold: float) -> np.ndarray:
    """余弦距离 + 聚类，返回标签数组。

    策略：
    - N <= _HIERARCHICAL_MAX: 直接层次聚类（精确）
    - N > _HIERARCHICAL_MAX: MiniBatchKMeans 粗聚类 → 递归拆分大子簇 → 层次聚类细分

    Args:
        features: (N, D) L2 归一化特征。
        threshold: 距离阈值，越小分组越多。

    Returns:
        (N,) int32 标签。
    """
    n = features.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.int32)

    if n <= _HIERARCHICAL_MAX:
        # 小样本直接层次聚类
        cosine_dist = _cosine_distance_matrix(features)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="precomputed",
            linkage="average",
        )
        return clustering.fit_predict(cosine_dist).astype(np.int32)

    # 大样本：MiniBatchKMeans 粗聚类 → 层次聚类细分
    logger.info("使用 MiniBatchKMeans + 层次聚类（样本数: %d）", n)
    from sklearn.cluster import MiniBatchKMeans

    # threshold 范围 [0.10, 0.80]，映射到粗聚类数
    coarse_ratio = 1.0 - (threshold - CLUSTER_THRESHOLD_MIN) / (CLUSTER_THRESHOLD_MAX - CLUSTER_THRESHOLD_MIN)
    min_clusters = max(2, n // 50)
    max_clusters = min(n // 5, 200)
    estimated_clusters = int(min_clusters + coarse_ratio * (max_clusters - min_clusters))
    estimated_clusters = max(1, min(estimated_clusters, n))

    kmeans = MiniBatchKMeans(
        n_clusters=estimated_clusters,
        random_state=42,
        batch_size=min(1024, n),
        n_init=3,
        max_iter=100,
    )
    labels = kmeans.fit_predict(features)

    # 递归拆分大子簇，直到所有子簇 <= _HIERARCHICAL_MAX
    _MAX_SPLIT_ROUNDS = 20
    changed = True
    while changed and _MAX_SPLIT_ROUNDS > 0:
        _MAX_SPLIT_ROUNDS -= 1
        changed = False
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_size = int(np.sum(cluster_mask))

            if cluster_size <= _HIERARCHICAL_MAX:
                continue

            # 子簇太大，用 MiniBatchKMeans 继续拆分
            changed = True
            cluster_features = features[cluster_mask]
            sub_k = max(2, cluster_size // (_HIERARCHICAL_MAX // 2))
            sub_kmeans = MiniBatchKMeans(
                n_clusters=sub_k,
                random_state=42,
                batch_size=min(1024, cluster_size),
                n_init=3,
            )
            sub_labels = sub_kmeans.fit_predict(cluster_features)
            base_label = int(labels.max()) + 1
            mask_indices = np.where(cluster_mask)[0]
            for j, idx in enumerate(mask_indices):
                labels[idx] = base_label + sub_labels[j]

    # 对每个子簇执行层次聚类细分
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_size = int(np.sum(cluster_mask))

        if cluster_size <= 1:
            continue

        cluster_features = features[cluster_mask]
        cosine_dist = _cosine_distance_matrix(cluster_features)

        sub_clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="precomputed",
            linkage="average",
        )
        sub_labels = sub_clustering.fit_predict(cosine_dist)

        base_label = int(labels.max()) + 1
        mask_indices = np.where(cluster_mask)[0]
        for j, idx in enumerate(mask_indices):
            labels[idx] = base_label + sub_labels[j]

    return labels.astype(np.int32)


def build_split_result(
    folder: Path,
    paths: list[Path],
    features: np.ndarray,
    labels: np.ndarray,
    use_smart_naming: bool = False,
) -> SplitResult:
    """根据聚类标签构建 SplitResult。

    Args:
        use_smart_naming: 是否使用 CLIP 智能命名。默认 False 以保持快速，
            仅在首次拆分（非 recluster）时建议启用。
    """
    groups: list[SplitGroup] = []
    unique_labels = sorted(set(labels.tolist()))

    namer = None
    if use_smart_naming:
        try:
            from .smart_naming import get_namer
            namer = get_namer()
        except ImportError:
            pass

    for group_id, label in enumerate(unique_labels):
        indices = [i for i, lb in enumerate(labels) if lb == label]
        group_paths = [paths[i] for i in indices]

        name = f"配色_{group_id + 1}"
        if namer and group_paths:
            try:
                name = namer.generate_name(group_paths)
            except Exception as e:
                logger.warning(f"智能命名失败: {e}")

        group = SplitGroup(
            id=group_id,
            name=name,
            image_paths=group_paths,
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
    """用缓存特征重新聚类（主线程同步调用）。

    注意：对 <=500 张图片的场景通常 <50ms，>500 张可能更久。
    不调用 smart_naming 以保持快速响应。
    """
    threshold = max(CLUSTER_THRESHOLD_MIN, min(CLUSTER_THRESHOLD_MAX, threshold))
    labels = cluster_images(result.features, threshold)
    return build_split_result(
        result.album_folder, result.image_paths, result.features, labels,
        use_smart_naming=False,
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
    
    # 批量创建文件夹
    for group in result.groups:
        folder_name = f"{album_name} {group.name}"
        folder_path, _ = unique_folder_path(parent, folder_name)
        folder_path.mkdir(parents=True, exist_ok=True)
        created_folders.append(folder_path)
    
    # 并行移动文件
    import concurrent.futures
    import os
    import threading

    # 定义进度计数器
    progress_count = 0
    progress_lock = threading.Lock()
    
    def update_progress():
        nonlocal progress_count
        with progress_lock:
            progress_count += 1
            if on_progress:
                on_progress(progress_count, total)
    
    # 收集所有文件移动任务（检测同名冲突并重命名）
    move_tasks = []
    for group, folder_path in zip(result.groups, created_folders):
        seen_names: set[str] = set()
        for img_path in group.image_paths:
            name = img_path.name
            if name in seen_names:
                # 同名冲突：添加序号后缀
                stem = img_path.stem
                suffix = img_path.suffix
                counter = 1
                while name in seen_names:
                    name = f"{stem}_{counter}{suffix}"
                    counter += 1
            seen_names.add(name)
            dest = folder_path / name
            move_tasks.append((img_path, dest))
    
    # 并行执行文件移动
    max_workers = min(8, os.cpu_count() or 4)
    move_errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 定义移动文件的函数
        def move_file(img_path, dest):
            try:
                if img_path.exists():
                    shutil.move(str(img_path), str(dest))
            except Exception as e:
                move_errors.append(f"{img_path.name}: {e}")
                logger.error(f"移动文件失败: {img_path} → {dest}: {e}")
            finally:
                update_progress()

        # 提交所有任务
        futures = [executor.submit(move_file, img_path, dest) for img_path, dest in move_tasks]
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"文件移动任务异常: {e}")

    if move_errors:
        logger.warning("拆分完成，但 %d 个文件移动失败", len(move_errors))

    # 原产品文件夹已清空，删除
    try:
        if album_folder.exists() and not any(album_folder.iterdir()):
            album_folder.rmdir()
    except OSError:
        pass  # 目录非空或权限不足，忽略

    # 记录拆分历史：原文件夹 + 所有子文件夹（单事务批量写入）
    history_db = get_split_history()
    split_time = datetime.now().isoformat()

    groups_data = []
    images_data = []
    for group, folder_path in zip(result.groups, created_folders):
        groups_data.append((group.id, group.name))
        for img_path in group.image_paths:
            images_data.append((group.id, str(img_path)))

    sub_folders = [
        (str(fp.resolve()), split_time, 1, len(g.image_paths), False)
        for fp, g in zip(created_folders, result.groups)
    ]

    history_db.add_split_history_batch(
        folder=str(album_folder.resolve()),
        split_time=split_time,
        group_count=len(result.groups),
        image_count=total,
        has_features=False,
        groups=groups_data,
        images=images_data,
        sub_folders=sub_folders,
    )

    return created_folders


def check_split_history(folder: Path) -> bool:
    """检查文件夹是否已经拆分过

    Args:
        folder: 相册文件夹路径

    Returns:
        bool: 是否已拆分过
    """
    history_db = get_split_history()
    # 使用 resolve() 标准化路径，避免大小写/尾部斜杠/相对路径导致匹配失败
    record = history_db.get_split_history(str(folder.resolve()))
    return record is not None

def extract_and_split(
    folder: Path,
    threshold: float = CLUSTER_THRESHOLD_DEFAULT,
    on_status: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    force: bool = False,
) -> SplitResult:
    """完整流水线：列出图片 → 加载模型 → 提取特征 → 聚类。

    Args:
        folder: 相册文件夹路径。
        threshold: 聚类距离阈值。
        on_status: 状态文字回调。
        on_progress: 特征提取进度回调 (当前, 总数)。
        force: 为 True 时忽略拆分历史，强制重新拆分。
    """
    # 检查是否已经拆分过（force 模式下跳过检查）
    if not force and check_split_history(folder):
        if on_status:
            on_status(f"文件夹 {folder.name} 已拆分过，跳过")
        images = list_images(folder)
        return SplitResult(
            album_folder=folder, groups=[], image_paths=images, features=np.empty((0, COMBINED_DIM)),
        )

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

    result = build_split_result(folder, images, features, labels, use_smart_naming=True)

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
    force: bool = False,
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
        force: 为 True 时忽略拆分历史，强制重新拆分。

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

        # 检查是否已经拆分过（force 模式下跳过检查）
        if not force and check_split_history(folder):
            if on_status:
                on_status(f"文件夹 {folder.name} 已拆分过，跳过")
            images = list_images(folder)
            result = SplitResult(
                album_folder=folder, groups=[], image_paths=images,
                features=np.empty((0, COMBINED_DIM)),
            )
            item = BatchScanItem(
                folder=folder, image_count=len(images), group_count=0,
                result=result, error=None,
            )
            results.append(item)
            if on_folder_done:
                on_folder_done(idx, total, item)
            continue

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
                result = build_split_result(folder, images, features, labels, use_smart_naming=True)
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
