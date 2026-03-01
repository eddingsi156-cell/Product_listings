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

logger = logging.getLogger(__name__)

from ..config import (
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

_instance: FeatureExtractor | None = None
_instance_lock = threading.Lock()


def get_extractor() -> FeatureExtractor:
    """获取全局单例 FeatureExtractor（线程安全）。"""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = FeatureExtractor()
    return _instance


class FeatureExtractor:
    """CLIP ViT-B/32 + HSV 颜色直方图特征提取。

    模型延迟加载：首次调用 ``load_model()`` 时才下载/加载权重。
    """

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False

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

        import open_clip

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

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return np.zeros(CLIP_DIM, dtype=np.float32)

        tensor = self._preprocess(img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            feat = self._model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        return feat.cpu().numpy().astype(np.float32).reshape(CLIP_DIM)

    def extract_clip_batch(
        self, paths: list[Path], batch_size: int = DEDUP_BATCH_SIZE,
    ) -> np.ndarray:
        """批量 CLIP 特征（自动分批），返回 (N, 512) float32。

        损坏的图片会降级为零向量，不会导致整个批次失败。
        """
        if not self._loaded:
            raise RuntimeError("必须先调用 load_model()")

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
                batch = torch.stack(tensors).to(self._device)
                try:
                    with torch.no_grad():
                        feats = self._model.encode_image(batch)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                except torch.cuda.OutOfMemoryError:
                    # GPU OOM → 仅当前批次降级到 CPU，不永久修改设备
                    logger.warning("GPU 显存不足，当前批次降级到 CPU 推理")
                    torch.cuda.empty_cache()
                    batch_cpu = batch.cpu()
                    with torch.no_grad():
                        feats = self._model.cpu().encode_image(batch_cpu)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                    # 恢复模型到 GPU，后续小批次仍可尝试使用 GPU
                    self._model = self._model.to(self._device)
                feats_np = feats.cpu().numpy().astype(np.float32)

                # 将失败的图片插回为零向量，保持与输入顺序一致
                if failed_indices:
                    result = np.zeros(
                        (len(chunk), CLIP_DIM), dtype=np.float32,
                    )
                    ok_idx = 0
                    for i in range(len(chunk)):
                        if i in failed_indices:
                            continue
                        result[i] = feats_np[ok_idx]
                        ok_idx += 1
                    all_feats.append(result)
                else:
                    all_feats.append(feats_np)
            else:
                # 整个 chunk 全部失败
                all_feats.append(
                    np.zeros((len(chunk), CLIP_DIM), dtype=np.float32)
                )

        if not all_feats:
            return np.empty((0, CLIP_DIM), dtype=np.float32)

        return np.vstack(all_feats)

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

        # 前景遮罩：排除过暗（黑色背景）和低饱和度+高亮度（白色背景）像素
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]
        fg_mask = (v_channel >= HSV_FG_V_MIN) & ~(
            (v_channel >= HSV_FG_V_MAX) & (s_channel < HSV_FG_S_MIN)
        )
        fg_mask = fg_mask.astype(np.uint8) * 255

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

        Args:
            paths: 图片路径列表。
            on_progress: 回调 (当前索引, 总数)。
        """
        n = len(paths)
        if n == 0:
            return np.empty((0, COMBINED_DIM), dtype=np.float32)

        # CLIP 批量推理
        clip_feats = self.extract_clip_batch(paths)  # (N, 512)

        # HSV 逐张（极快，无需批处理）
        hsv_feats = np.empty((n, HSV_DIM), dtype=np.float32)
        for i, p in enumerate(paths):
            hsv_feats[i] = self.extract_hsv(p)
            if on_progress:
                on_progress(i + 1, n)

        # 拼接 CLIP(512 维) + HSV×1.5(96 维) 后 L2 归一化。
        # 归一化前 CLIP 与 HSV 的 L2 范数比约 1:1.5，归一化后 HSV 在
        # 余弦距离中的有效贡献约 10-15%，足以区分同款不同色。
        combined = np.concatenate(
            [clip_feats, hsv_feats * HSV_WEIGHT], axis=1,
        )  # (N, 608)
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        combined /= norms

        return combined
