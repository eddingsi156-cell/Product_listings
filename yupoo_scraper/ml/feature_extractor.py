"""CLIP + HSV 特征提取器 — 单例模式，延迟加载模型"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image

# 提前导入 open_clip，避免在 load_model 方法中导入导致的阻塞
import open_clip

# 提前导入特征缓存，避免在方法中导入导致的阻塞
from .feature_cache import get_cache

from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

from ..config import (
    CLIP_BATCH_SIZE_CPU,
    CLIP_BATCH_SIZE_GPU,
    CLIP_DIM,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    COMBINED_DIM,
    DEDUP_BATCH_SIZE,
    HSV_BINS,
    HSV_DIM,
    HSV_FG_S_MIN,
    HSV_FG_V_MAX,
    HSV_FG_V_MIN,
    HSV_RANGES,
    HSV_WEIGHT,
)

_instance: 'FeatureExtractor' | None = None
_instance_lock = threading.Lock()


def get_extractor() -> 'FeatureExtractor':
    """获取全局单例 FeatureExtractor（线程安全）。"""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = FeatureExtractor()
    return _instance


# DataLoader 预加载阈值：图片数 >= 此值时启用多 worker 预加载
_DATALOADER_THRESHOLD = 32
_DATALOADER_NUM_WORKERS = 4


class _ImageDataset(Dataset):
    """轻量 Dataset，用于 DataLoader 多进程预加载和预处理图片。"""

    def __init__(self, paths: list[Path], preprocess):
        self._paths = paths
        self._preprocess = preprocess

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx):
        path = self._paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self._preprocess(img)
            return tensor, idx, True   # (preprocessed_tensor, original_index, success)
        except Exception:
            # 返回零张量 + 失败标记
            return torch.zeros(3, 224, 224), idx, False


class FeatureExtractor:
    """CLIP ViT-B/32 + HSV 颜色直方图特征提取。

    模型延迟加载：首次调用 ``load_model()`` 时才下载/加载权重。
    """

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False
        self._batch_size: int = (
            CLIP_BATCH_SIZE_GPU if self._device == "cuda" else CLIP_BATCH_SIZE_CPU
        )

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    @property
    def model(self):
        return self._model

    # ── 模型加载 ───────────────────────────────────────────────

    def load_model(
        self, on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """加载 CLIP 模型（首次调用时下载权重）。"""
        if self._loaded:
            return

        if on_progress:
            on_progress("正在加载 CLIP 模型...")

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED,
        )
        model = model.to(self._device).eval()

        self._model = model
        self._preprocess = preprocess
        self._loaded = True

        if on_progress:
            on_progress("CLIP 模型加载完成")

    def unload_model(self) -> None:
        """释放 CLIP 模型和 GPU 显存。"""
        if self._model is not None:
            del self._model
            self._model = None
            self._preprocess = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── CLIP 特征 ──────────────────────────────────────────────

    def extract_clip(self, path: Path) -> np.ndarray:
        """单张图片 CLIP 特征，返回 (512,) float32。"""
        if not self._loaded:
            raise RuntimeError("必须先调用 load_model()")

        # 尝试从缓存获取
        cache = get_cache()
        cached_feat = cache.get(path)
        if cached_feat is not None and cached_feat.shape == (CLIP_DIM,):
            return cached_feat

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return np.zeros(CLIP_DIM, dtype=np.float32)

        tensor = self._preprocess(img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            feat = self._model.encode_image(tensor)
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)

        feat_np = feat.cpu().numpy().astype(np.float32).reshape(CLIP_DIM)
        # 缓存特征
        cache.set(path, feat_np)
        return feat_np

    def extract_clip_batch(
        self, paths: list[Path], batch_size: int | None = None,
    ) -> np.ndarray:
        """批量 CLIP 特征（自动分批），返回 (N, 512) float32。

        Args:
            paths: 图片路径列表。
            batch_size: 推理批大小。None 时使用自适应值
                        （GPU=CLIP_BATCH_SIZE_GPU, CPU=CLIP_BATCH_SIZE_CPU）。
                        OOM 时自动减半并记忆。

        损坏的图片会降级为零向量，不会导致整个批次失败。
        """
        if not self._loaded:
            raise RuntimeError("必须先调用 load_model()")

        if batch_size is None:
            batch_size = self._batch_size

        cache = get_cache()

        all_feats: list[np.ndarray] = []
        remaining_paths: list[Path] = []
        cached_feats: list[np.ndarray] = []
        cached_indices: list[int] = []

        # 首先检查缓存
        for i, path in enumerate(paths):
            cached_feat = cache.get(path)
            if cached_feat is not None and cached_feat.shape == (CLIP_DIM,):
                cached_feats.append(cached_feat)
                cached_indices.append(i)
            else:
                remaining_paths.append(path)

        # 处理缓存未命中的路径
        if remaining_paths:
            use_dataloader = (
                len(remaining_paths) >= _DATALOADER_THRESHOLD
                and _DATALOADER_NUM_WORKERS > 0
            )

            if use_dataloader:
                # DataLoader 多 worker 预加载 — 图片 IO 和预处理并行化
                all_feats = self._extract_with_dataloader(
                    remaining_paths, batch_size, cache,
                )
            else:
                all_feats = self._extract_sequential(
                    remaining_paths, batch_size, cache,
                )

        # 合并缓存命中和未命中的结果
        final_feats = np.zeros((len(paths), CLIP_DIM), dtype=np.float32)
        
        # 填充缓存命中的特征
        for i, idx in enumerate(cached_indices):
            final_feats[idx] = cached_feats[i]
        
        # 填充未缓存的特征
        if remaining_paths:
            non_cached_feats = np.vstack(all_feats) if all_feats else np.empty((0, CLIP_DIM), dtype=np.float32)
            cached_set = set(cached_indices)
            non_cached_idx = 0
            for i in range(len(paths)):
                if i not in cached_set:
                    if non_cached_idx < len(non_cached_feats):
                        final_feats[i] = non_cached_feats[non_cached_idx]
                        non_cached_idx += 1

        # 批量操作结束，刷新缓存索引
        cache.flush()

        return final_feats

    # ── CLIP 批量提取内部方法 ───────────────────────────────────

    def _extract_sequential(
        self, paths: list[Path], batch_size: int, cache,
    ) -> list[np.ndarray]:
        """顺序加载图片并推理（少量图片时使用）。"""
        all_feats: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            tensors = []
            failed_indices: list[int] = []

            for i, p in enumerate(chunk):
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(self._preprocess(img))
                except Exception:
                    failed_indices.append(i)

            if tensors:
                feats_np = self._infer_batch(torch.stack(tensors), batch_size)
                all_feats.append(
                    self._merge_with_failures(chunk, feats_np, failed_indices, cache)
                )
            else:
                all_feats.append(
                    np.zeros((len(chunk), CLIP_DIM), dtype=np.float32)
                )
        return all_feats

    def _extract_with_dataloader(
        self, paths: list[Path], batch_size: int, cache,
    ) -> list[np.ndarray]:
        """使用 DataLoader 多 worker 并行预加载图片，GPU 推理流水线化。"""
        dataset = _ImageDataset(paths, self._preprocess)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=_DATALOADER_NUM_WORKERS,
            pin_memory=(self._device == "cuda"),
            persistent_workers=False,
        )

        # 按原始顺序收集结果
        result = np.zeros((len(paths), CLIP_DIM), dtype=np.float32)
        for batch_tensors, indices, success_flags in loader:
            ok_mask = success_flags.bool()
            ok_tensors = batch_tensors[ok_mask]
            ok_indices = indices[ok_mask].numpy()

            if len(ok_tensors) > 0:
                feats_np = self._infer_batch(ok_tensors, batch_size)
                for j, orig_idx in enumerate(ok_indices):
                    result[orig_idx] = feats_np[j]
                    cache.set(paths[orig_idx], feats_np[j])

        # 返回为单元素 list 以兼容 all_feats 格式
        return [result]

    def _infer_batch(self, tensors: torch.Tensor, batch_size: int) -> np.ndarray:
        """GPU/CPU 推理，含 OOM 降级逻辑。返回 (N, CLIP_DIM) float32。"""
        batch = tensors.to(self._device)
        try:
            with torch.no_grad():
                feats = self._model.encode_image(batch)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        except torch.cuda.OutOfMemoryError:
            new_bs = max(1, batch_size // 2)
            logger.warning(
                "GPU 显存不足，批大小 %d → %d，当前批次降级到 CPU",
                batch_size, new_bs,
            )
            self._batch_size = new_bs
            torch.cuda.empty_cache()
            # 将模型移到 CPU 并保持引用一致
            self._model = self._model.cpu()
            batch_cpu = batch.cpu()
            with torch.no_grad():
                feats = self._model.encode_image(batch_cpu)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            # 尝试恢复模型到 GPU
            if self._device != "cpu":
                try:
                    self._model = self._model.to(self._device)
                except torch.cuda.OutOfMemoryError:
                    logger.warning("恢复模型到 GPU 失败，永久降级到 CPU")
                    torch.cuda.empty_cache()
                    self._device = "cpu"
                    self._batch_size = CLIP_BATCH_SIZE_CPU
        return feats.cpu().numpy().astype(np.float32)

    @staticmethod
    def _merge_with_failures(
        chunk: list[Path], feats_np: np.ndarray,
        failed_indices: list[int], cache,
    ) -> np.ndarray:
        """将推理结果与失败项合并，保持原始顺序。"""
        if not failed_indices:
            for i, path in enumerate(chunk):
                cache.set(path, feats_np[i])
            return feats_np

        result = np.zeros((len(chunk), CLIP_DIM), dtype=np.float32)
        failed_set = set(failed_indices)
        ok_idx = 0
        for i in range(len(chunk)):
            if i in failed_set:
                continue
            result[i] = feats_np[ok_idx]
            cache.set(chunk[i], feats_np[ok_idx])
            ok_idx += 1
        return result

    # ── HSV 颜色直方图 ─────────────────────────────────────────

    @staticmethod
    def extract_hsv(path: Path) -> np.ndarray:
        """单张图片 HSV 颜色直方图（含前景遮罩），返回 (96,) float32。"""
        # cv2.imread 不支持 Windows Unicode 路径，用 imdecode 代替
        buf = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return np.zeros(HSV_DIM, dtype=np.float32)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # 前景遮罩：仅排除背景像素（纯白底/纯黑底），保留产品本身的深色像素
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]
        # 排除纯白背景：高亮度 + 低饱和度
        is_white_bg = (v_channel >= HSV_FG_V_MAX) & (s_channel < HSV_FG_S_MIN)
        # 排除纯黑背景：极暗 + 低饱和度（有颜色的深色像素保留）
        is_black_bg = (v_channel < HSV_FG_V_MIN) & (s_channel < HSV_FG_S_MIN)
        fg_mask = (~is_white_bg & ~is_black_bg).astype(np.uint8) * 255

        # 如果前景像素太少（<5%），回退到无遮罩计算（可能不是产品图）
        if np.count_nonzero(fg_mask) < 0.05 * fg_mask.size:
            fg_mask = None

        hist = cv2.calcHist(
            [hsv], [0, 1, 2], fg_mask,
            list(HSV_BINS), HSV_RANGES[0] + HSV_RANGES[1] + HSV_RANGES[2],
        )
        hist = hist.flatten().astype(np.float32)

        # L2 归一化
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm

        return hist

    # ── 组合特征 ───────────────────────────────────────────────

    def extract_combined(self, path: Path) -> np.ndarray:
        """单张图片组合特征（CLIP + HSV 加权），返回 (608,) float32。"""
        clip_feat = self.extract_clip(path)
        hsv_feat = self.extract_hsv(path) * HSV_WEIGHT
        combined = np.concatenate([clip_feat, hsv_feat])
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined /= norm
        return combined

    def extract_combined_batch(
        self,
        paths: list[Path],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """批量组合特征，返回 (N, 608) float32。

        优先从缓存读取 combined 特征，缓存未命中的才走 CLIP+HSV 提取流程。

        Args:
            paths: 图片路径列表。
            on_progress: 回调 (当前索引, 总数)。
        """
        n = len(paths)
        if n == 0:
            return np.empty((0, COMBINED_DIM), dtype=np.float32)

        cache = get_cache()

        # ── 第一步：检查 combined 缓存 ──
        combined_result = np.empty((n, COMBINED_DIM), dtype=np.float32)
        uncached_indices: list[int] = []
        uncached_paths: list[Path] = []

        for i, path in enumerate(paths):
            cached = cache.get(path, namespace="combined")
            if cached is not None and cached.shape == (COMBINED_DIM,):
                combined_result[i] = cached
            else:
                uncached_indices.append(i)
                uncached_paths.append(path)

        # 全部命中缓存
        if not uncached_paths:
            if on_progress:
                on_progress(n, n)
            return combined_result

        # ── 第二步：对缓存未命中的图片提取 CLIP + HSV ──
        import concurrent.futures

        n_uncached = len(uncached_paths)
        # 总进度 = CLIP 提取(n_uncached) + HSV 提取(n_uncached)
        total_steps = n_uncached * 2
        progress_count = 0
        progress_lock = threading.Lock()

        def update_progress():
            nonlocal progress_count
            with progress_lock:
                progress_count += 1
                if on_progress:
                    on_progress(progress_count, total_steps)

        # 提取 CLIP 特征（内部有自己的 CLIP 缓存）
        clip_feats = self.extract_clip_batch(uncached_paths)
        for _ in range(n_uncached):
            update_progress()

        # 并行提取 HSV 特征
        hsv_feats = np.empty((n_uncached, HSV_DIM), dtype=np.float32)

        def process_hsv(i, path):
            try:
                feat = self.extract_hsv(path)
                hsv_feats[i] = feat
            except Exception as e:
                logger.warning(f"提取 HSV 特征失败: {e}")
                hsv_feats[i] = np.zeros(HSV_DIM, dtype=np.float32)
            finally:
                update_progress()

        import os
        max_workers = min(16, os.cpu_count() or 8)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_hsv, i, p) for i, p in enumerate(uncached_paths)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"HSV 特征提取任务失败: {e}")

        # 拼接 CLIP(512 维) + HSV×1.5(96 维) 后 L2 归一化
        combined_uncached = np.concatenate(
            [clip_feats, hsv_feats * HSV_WEIGHT], axis=1,
        )  # (n_uncached, 608)
        norms = np.linalg.norm(combined_uncached, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        combined_uncached /= norms

        # 写入结果并缓存 combined 特征
        for j, orig_idx in enumerate(uncached_indices):
            combined_result[orig_idx] = combined_uncached[j]
            cache.set(uncached_paths[j], combined_uncached[j], namespace="combined")

        cache.flush()
        return combined_result
