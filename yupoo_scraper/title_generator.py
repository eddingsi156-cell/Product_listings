"""标题生成器 — CLIP zero-shot 产品分类 + 随机词池组合"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)

from .config import (
    CATEGORY_PROMPTS,
    CLIP_MODEL_NAME,
    DETAIL_IMAGE_MAX,
    MAIN_IMAGE_MAX,
    TITLE_PREFIXES,
    TITLE_STYLES,
    TITLE_SUFFIXES,
)
from .config import IMAGE_EXTS
from .ml.feature_extractor import get_extractor


@dataclass
class ProductInfo:
    """待上架产品信息。"""

    folder: Path
    title: str = ""
    price: float = 0.0
    stock: int = 0
    main_images: list[Path] = field(default_factory=list)
    detail_images: list[Path] = field(default_factory=list)


class TitleGenerator:
    """基于 CLIP zero-shot 分类的标题生成器。"""

    def __init__(self) -> None:
        self._text_features: np.ndarray | None = None
        self._category_keys: list[str] = []

    def _ensure_model(
        self, on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """确保 CLIP 模型已加载，并预计算类别文本特征。"""
        ext = get_extractor()
        if not ext.loaded:
            ext.load_model(on_progress)

        if self._text_features is not None:
            return

        import open_clip

        self._category_keys = list(CATEGORY_PROMPTS.keys())
        prompts = [CATEGORY_PROMPTS[k][0] for k in self._category_keys]

        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        tokens = tokenizer(prompts).to(ext.device)

        with torch.no_grad():
            text_feats = ext.model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        self._text_features = text_feats.cpu().numpy().astype(np.float32)

    def classify_product(self, image_paths: list[Path]) -> str:
        """对产品图片做 zero-shot 分类，返回中文类别名。

        取所有图片的 CLIP 特征均值，与预设类别文本特征计算余弦相似度。
        """
        self._ensure_model()
        ext = get_extractor()

        if not image_paths:
            return "商品"

        # 取最多 5 张图做分类（避免过多图片拖慢速度）
        sample = image_paths[:5]
        clip_feats = ext.extract_clip_batch(sample)  # (N, 512)
        avg_feat = clip_feats.mean(axis=0, keepdims=True)  # (1, 512)
        norm = np.linalg.norm(avg_feat)
        if norm < 1e-8:
            # 所有图片特征提取失败（全零向量），回退为通用分类
            logger.warning("所有采样图片特征提取失败，使用默认分类")
            return "商品"
        avg_feat /= norm

        # 余弦相似度
        sims = avg_feat @ self._text_features.T  # (1, num_categories)
        best_idx = int(sims.argmax())
        best_key = self._category_keys[best_idx]

        return CATEGORY_PROMPTS[best_key][1]

    def generate_title(self, image_paths: list[Path]) -> str:
        """生成一个随机组合标题：前缀 + 风格 + 类别 + 后缀。"""
        category = self.classify_product(image_paths)
        prefix = random.choice(TITLE_PREFIXES)
        style = random.choice(TITLE_STYLES)
        suffix = random.choice(TITLE_SUFFIXES)
        return f"{prefix}{style}{category}{suffix}"

    def batch_generate(
        self,
        product_folders: list[Path],
        price: float = 0.0,
        stock: int = 0,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[ProductInfo]:
        """批量为产品文件夹生成标题和图片列表。

        Args:
            product_folders: 每个文件夹包含一个产品的图片。
            price: 统一价格。
            stock: 统一库存。
            on_progress: 回调 (当前索引, 总数)。

        Returns:
            ProductInfo 列表。
        """
        self._ensure_model()
        ext = get_extractor()
        results: list[ProductInfo] = []
        total = len(product_folders)

        for i, folder in enumerate(product_folders):
            images = sorted(
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )

            if not images:
                # 空文件夹也创建占位，保持与输入列表的索引对应
                results.append(ProductInfo(folder=folder, title="(空文件夹)"))
            else:
                title = self.generate_title(images)
                main_imgs = images[:MAIN_IMAGE_MAX]
                detail_imgs = images[MAIN_IMAGE_MAX:MAIN_IMAGE_MAX + DETAIL_IMAGE_MAX]

                results.append(ProductInfo(
                    folder=folder,
                    title=title,
                    price=price,
                    stock=stock,
                    main_images=main_imgs,
                    detail_images=detail_imgs,
                ))

            if on_progress:
                on_progress(i + 1, total)

        return results
