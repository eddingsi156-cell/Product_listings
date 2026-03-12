"""测试特征缓存功能"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from yupoo_scraper.ml.feature_cache import FeatureCache, get_cache
from yupoo_scraper.image_processor import list_images
from yupoo_scraper.config import DEFAULT_DOWNLOAD_DIR
import numpy as np

def test_cache_basic():
    """测试缓存的基本功能"""
    print("=== 测试缓存基本功能 ===")
    
    # 创建缓存实例
    cache = FeatureCache()
    print(f"缓存目录: {cache.cache_dir}")
    
    # 选择一个有图片的文件夹
    test_folder = None
    for folder in DEFAULT_DOWNLOAD_DIR.iterdir():
        if folder.is_dir():
            test_folder = folder
            break
    
    if not test_folder:
        print("错误：没有找到测试文件夹")
        return False
    
    # 获取测试图片
    images = list_images(test_folder)
    if not images:
        print("错误：没有找到测试图片")
        return False
    
    test_image = images[0]
    print(f"测试图片: {test_image}")
    
    # 测试设置和获取缓存
    try:
        # 创建测试特征
        test_features = np.random.rand(512).astype(np.float32)
        
        # 设置缓存
        print("设置缓存...")
        cache.set(test_image, test_features)
        print("设置缓存成功")
        
        # 获取缓存
        print("获取缓存...")
        retrieved_features = cache.get(test_image)
        print(f"获取缓存成功: {retrieved_features is not None}")
        
        if retrieved_features is not None:
            print(f"特征形状: {retrieved_features.shape}")
            print(f"特征值: {retrieved_features[:5]}")
        
        # 测试缓存大小
        print(f"缓存大小: {cache.size()}")
        
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_global_cache():
    """测试全局缓存实例"""
    print("\n=== 测试全局缓存实例 ===")
    try:
        cache1 = get_cache()
        cache2 = get_cache()
        print(f"全局缓存实例是否相同: {cache1 is cache2}")
        print(f"缓存目录: {cache1.cache_dir}")
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试特征缓存...")
    
    # 测试基本功能
    basic_ok = test_cache_basic()
    print(f"\n基本功能测试: {'通过' if basic_ok else '失败'}")
    
    # 测试全局缓存
    global_ok = test_global_cache()
    print(f"全局缓存测试: {'通过' if global_ok else '失败'}")
    
    print("\n测试完成!")
