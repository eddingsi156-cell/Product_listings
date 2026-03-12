"""直接测试特征提取器的 load_model 方法"""

import sys
import time
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

# 直接导入必要的模块
print("导入基本模块...")
import torch
print(f"Torch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

print("导入 open_clip...")
import open_clip
print(f"OpenCLIP 版本: {open_clip.__version__}")

print("导入配置...")
from yupoo_scraper.config import CLIP_MODEL_NAME, CLIP_PRETRAINED
print(f"模型名称: {CLIP_MODEL_NAME}")
print(f"预训练权重: {CLIP_PRETRAINED}")

print("创建特征提取器...")
from yupoo_scraper.ml.feature_extractor import FeatureExtractor

# 创建实例
extractor = FeatureExtractor()
print(f"特征提取器实例: {extractor}")
print(f"设备: {extractor.device}")
print(f"模型是否已加载: {extractor.loaded}")

print("开始加载模型...")
start_time = time.time()

try:
    # 直接调用 open_clip.create_model_and_transforms
    print("直接调用 open_clip.create_model_and_transforms...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    print("模型创建成功")
    
    print("移动模型到设备...")
    model = model.to(extractor.device).eval()
    print("移动成功")
    
    # 更新提取器状态
    extractor._model = model
    extractor._preprocess = preprocess
    extractor._loaded = True
    
    end_time = time.time()
    print(f"模型加载完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"模型加载状态: {extractor.loaded}")
    
    print("测试完成!")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
