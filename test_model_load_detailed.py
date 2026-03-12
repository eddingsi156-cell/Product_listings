"""详细测试模型加载功能"""

import sys
import time
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from yupoo_scraper.ml.feature_extractor import get_extractor
from yupoo_scraper.config import CLIP_MODEL_NAME, CLIP_PRETRAINED

def test_model_load():
    """详细测试模型加载功能"""
    print("=== 详细测试模型加载功能 ===")
    print(f"模型名称: {CLIP_MODEL_NAME}")
    print(f"预训练权重: {CLIP_PRETRAINED}")
    
    try:
        # 获取特征提取器实例
        extractor = get_extractor()
        print(f"特征提取器实例创建成功")
        print(f"设备: {extractor.device}")
        print(f"模型是否已加载: {extractor.loaded}")
        
        # 尝试加载模型
        print("开始加载 CLIP 模型...")
        start_time = time.time()
        
        # 直接导入 open_clip 并测试
        print("测试 open_clip 导入...")
        import open_clip
        print("open_clip 导入成功")
        
        print("测试模型创建...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED,
        )
        print("模型创建成功")
        
        print("测试模型移动到设备...")
        model = model.to(extractor.device).eval()
        print("模型移动到设备成功")
        
        end_time = time.time()
        print(f"模型加载完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"模型类型: {type(model)}")
        
        # 更新提取器状态
        extractor._model = model
        extractor._preprocess = preprocess
        extractor._loaded = True
        
        print(f"特征提取器状态更新成功: {extractor.loaded}")
        
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始详细测试模型加载...")
    
    # 测试模型加载
    load_ok = test_model_load()
    print(f"\n模型加载测试: {'通过' if load_ok else '失败'}")
    
    print("\n测试完成!")
