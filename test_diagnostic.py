"""诊断脚本：检查拆分功能的问题"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from yupoo_scraper.image_processor import list_images
from yupoo_scraper.ml.splitter import batch_extract_and_split
from yupoo_scraper.config import DEFAULT_DOWNLOAD_DIR

def test_list_images():
    """测试 list_images 函数"""
    print("=== 测试 list_images 函数 ===")
    # 选择一个有图片的文件夹
    test_folder = None
    for folder in DEFAULT_DOWNLOAD_DIR.iterdir():
        if folder.is_dir():
            test_folder = folder
            break
    
    if not test_folder:
        print("错误：没有找到测试文件夹")
        return False
    
    print(f"测试文件夹: {test_folder}")
    try:
        images = list_images(test_folder)
        print(f"找到 {len(images)} 张图片:")
        for img in images:
            print(f"  - {img.name}")
        return len(images) > 0
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_feature_extraction():
    """测试特征提取功能"""
    print("\n=== 测试特征提取功能 ===")
    # 选择一个有图片的文件夹
    test_folders = []
    for folder in DEFAULT_DOWNLOAD_DIR.iterdir():
        if folder.is_dir():
            test_folders.append(folder)
        if len(test_folders) >= 1:
            break
    
    if not test_folders:
        print("错误：没有找到测试文件夹")
        return False
    
    print(f"测试文件夹: {[f.name for f in test_folders]}")
    
    try:
        # 先测试模型加载
        print("测试模型加载...")
        from yupoo_scraper.ml.feature_extractor import get_extractor
        extractor = get_extractor()
        print(f"特征提取器实例: {extractor}")
        print(f"设备: {extractor.device}")
        
        print("加载模型...")
        extractor.load_model(on_progress=lambda msg: print(f"模型加载进度: {msg}"))
        print(f"模型加载完成: {extractor.loaded}")
        
        # 测试单张图片特征提取
        test_folder = test_folders[0]
        images = list_images(test_folder)
        if images:
            test_image = images[0]
            print(f"测试单张图片特征提取: {test_image.name}")
            clip_feat = extractor.extract_clip(test_image)
            print(f"CLIP 特征形状: {clip_feat.shape}")
            hsv_feat = extractor.extract_hsv(test_image)
            print(f"HSV 特征形状: {hsv_feat.shape}")
            combined_feat = extractor.extract_combined(test_image)
            print(f"组合特征形状: {combined_feat.shape}")
        
        # 测试批量特征提取
        print("测试批量特征提取...")
        if images:
            batch_feats = extractor.extract_combined_batch(images)
            print(f"批量特征形状: {batch_feats.shape}")
        
        # 测试聚类
        print("测试聚类...")
        from yupoo_scraper.ml.splitter import cluster_images
        if images and len(images) > 1:
            labels = cluster_images(batch_feats, 0.35)
            print(f"聚类标签: {labels}")
            print(f"唯一标签数: {len(set(labels))}")
        
        # 测试完整流程
        print("测试完整流程...")
        from yupoo_scraper.ml.splitter import extract_and_split
        result = extract_and_split(
            test_folder,
            threshold=0.35,
            on_status=lambda msg: print(f"流程状态: {msg}"),
            on_progress=lambda current, total: print(f"流程进度: {current}/{total}")
        )
        print(f"拆分结果: {len(result.groups)} 个分组")
        for i, group in enumerate(result.groups):
            print(f"  分组 {i+1}: {group.name}, {len(group.image_paths)} 张图片")
        
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始诊断...")
    
    # 测试 list_images
    list_images_ok = test_list_images()
    print(f"\nlist_images 测试: {'通过' if list_images_ok else '失败'}")
    
    # 测试特征提取
    feature_extraction_ok = test_feature_extraction()
    print(f"特征提取测试: {'通过' if feature_extraction_ok else '失败'}")
    
    print("\n诊断完成!")
