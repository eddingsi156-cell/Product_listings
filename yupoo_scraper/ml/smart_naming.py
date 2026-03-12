"""智能命名模块 - 基于 CLIP 模型自动生成分组名称"""

import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

from ..config import CATEGORY_PROMPTS
from .feature_extractor import get_extractor


class SmartNamer:
    """智能命名器，基于 CLIP 模型自动生成分组名称。"""

    def __init__(self):
        self.extractor = get_extractor()
        self.category_prompts = CATEGORY_PROMPTS
        self.color_prompts = [
            "红色", "蓝色", "绿色", "黑色", "白色", "灰色", "棕色", "紫色",
            "粉色", "橙色", "黄色", "青色", "金色", "银色", "透明"
        ]
        self.style_prompts = [
            "休闲", "正式", "运动", "时尚", "复古", "简约", "奢华", "街头"
        ]
        # 缓存 text features，避免每次调用都重新 tokenize + encode
        self._text_feat_cache: dict[str, np.ndarray] = {}

    def _get_image_features(self, image_path: Path) -> Optional[np.ndarray]:
        """获取图片特征。"""
        try:
            return self.extractor.extract_clip(image_path)
        except Exception as e:
            logger.warning(f"提取图片特征失败: {e}")
            return None

    def _get_text_features(self, texts: List[str]) -> np.ndarray:
        """获取文本特征（带缓存）。"""
        cache_key = "\n".join(texts)
        if cache_key in self._text_feat_cache:
            return self._text_feat_cache[cache_key]

        if not self.extractor.loaded:
            self.extractor.load_model()

        import open_clip
        from ..config import CLIP_MODEL_NAME
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        tokens = tokenizer(texts).to(self.extractor.device)

        with torch.no_grad():
            text_features = self.extractor.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        result = text_features.cpu().numpy()
        self._text_feat_cache[cache_key] = result
        return result

    def _predict_category(self, image_features: np.ndarray) -> Optional[str]:
        """预测产品类别。"""
        category_items = list(self.category_prompts.items())
        text_prompts = [prompt[0] for _key, prompt in category_items]
        text_features = self._get_text_features(text_prompts)

        similarities = image_features @ text_features.T
        best_idx = np.argmax(similarities)

        if similarities[best_idx] > 0.2:  # 阈值
            return category_items[best_idx][1][1]
        return None

    def _predict_color(self, image_features: np.ndarray) -> Optional[str]:
        """预测产品颜色。"""
        text_prompts = [f"{color}的产品" for color in self.color_prompts]
        text_features = self._get_text_features(text_prompts)

        similarities = image_features @ text_features.T
        best_idx = np.argmax(similarities)
        
        if similarities[best_idx] > 0.15:  # 阈值
            return self.color_prompts[best_idx]
        return None

    def _predict_style(self, image_features: np.ndarray) -> Optional[str]:
        """预测产品风格。"""
        text_prompts = [f"{style}风格的产品" for style in self.style_prompts]
        text_features = self._get_text_features(text_prompts)

        similarities = image_features @ text_features.T
        best_idx = np.argmax(similarities)
        
        if similarities[best_idx] > 0.15:  # 阈值
            return self.style_prompts[best_idx]
        return None

    def generate_name(self, image_paths: List[Path]) -> str:
        """为一组图片生成名称。"""
        if not image_paths:
            return "未命名"

        # 加载模型
        if not self.extractor.loaded:
            self.extractor.load_model()

        # 提取所有图片的特征
        features = []
        for path in image_paths[:5]:  # 只使用前5张图片
            feat = self._get_image_features(path)
            if feat is not None:
                features.append(feat)

        if not features:
            return "未命名"

        # 计算平均特征并重新归一化
        avg_features = np.mean(features, axis=0)
        norm = np.linalg.norm(avg_features)
        if norm > 0:
            avg_features = avg_features / norm

        # 预测类别、颜色、风格
        category = self._predict_category(avg_features)
        color = self._predict_color(avg_features)
        style = self._predict_style(avg_features)

        # 生成名称
        name_parts = []
        if category:
            name_parts.append(category)
        if color:
            name_parts.append(color)
        if style:
            name_parts.append(style)

        if name_parts:
            return " ".join(name_parts)
        else:
            return "未命名"


# 全局命名器实例（双重检查锁保证线程安全）
_namer_instance: Optional[SmartNamer] = None
_namer_lock = threading.Lock()


def get_namer() -> SmartNamer:
    """获取全局命名器实例（线程安全）。"""
    global _namer_instance
    if _namer_instance is None:
        with _namer_lock:
            if _namer_instance is None:
                _namer_instance = SmartNamer()
    return _namer_instance
