"""测试模型加载功能"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from yupoo_scraper.ml.feature_extractor import get_extractor

def test_model_load():
    """测试模型加载功能"""
    print("=== 测试模型加载功能 ===")
    
    try:
        # 获取特征提取器实例
        extractor = get_extractor()
        print(f"特征提取器实例创建成功")
        print(f"设备: {extractor.device}")
        print(f"模型是否已加载: {extractor.loaded}")
        
        # 尝试加载模型
        print("开始加载 CLIP 模型...")
        extractor.load_model(
            on_progress=lambda msg: print(f"进度: {msg}")
        )
        print(f"模型加载成功: {extractor.loaded}")
        print(f"模型: {extractor.model}")
        
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试模型加载...")
    
    # 测试模型加载
    load_ok = test_model_load()
    print(f"\n模型加载测试: {'通过' if load_ok else '失败'}")
    
    print("\n测试完成!")
