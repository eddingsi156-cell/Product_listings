"""测试拆分功能的性能"""

import time
import tempfile
from pathlib import Path
import numpy as np
import logging
from yupoo_scraper.ml.splitter import extract_and_split, batch_extract_and_split
from yupoo_scraper.image_processor import list_images

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建测试图片
def create_test_images(folder, count):
    """创建测试图片文件"""
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (folder / f"test_{i}.jpg").write_text(f"test{i}")

# 测试单个文件夹的拆分性能
def test_single_folder_performance():
    """测试单个文件夹的拆分性能"""
    print("=== 测试单个文件夹的拆分性能 ===")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试文件夹和图片
            test_folder = Path(tmpdir) / "test_album"
            create_test_images(test_folder, 50)  # 创建50张测试图片
            
            # 检查图片是否创建成功
            images = list_images(test_folder)
            print(f"创建的图片数量: {len(images)}")
            
            if not images:
                print("错误: 没有找到图片")
                return
            
            # 记录开始时间
            start_time = time.time()
            
            # 执行拆分
            logger.info("开始执行拆分")
            result = extract_and_split(test_folder, on_status=lambda s: logger.info(s))
            
            # 记录结束时间
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"文件夹: {test_folder.name}")
            print(f"图片数量: {len(result.image_paths)}")
            print(f"分组数量: {len(result.groups)}")
            print(f"总耗时: {total_time:.2f} 秒")
            print(f"平均每张图片耗时: {total_time / len(result.image_paths):.4f} 秒")
            print()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

# 测试批量文件夹的拆分性能
def test_batch_folder_performance():
    """测试批量文件夹的拆分性能"""
    print("=== 测试批量文件夹的拆分性能 ===")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建多个测试文件夹
            test_folders = []
            for i in range(5):  # 创建5个测试文件夹
                folder = Path(tmpdir) / f"test_album_{i}"
                create_test_images(folder, 20)  # 每个文件夹20张图片
                test_folders.append(folder)
            
            # 检查图片是否创建成功
            total_images = 0
            for folder in test_folders:
                images = list_images(folder)
                total_images += len(images)
                print(f"文件夹 {folder.name} 图片数量: {len(images)}")
            
            if total_images == 0:
                print("错误: 没有找到图片")
                return
            
            # 记录开始时间
            start_time = time.time()
            
            # 执行批量拆分
            logger.info("开始执行批量拆分")
            results = batch_extract_and_split(
                test_folders, 
                threshold=0.5,
                on_status=lambda s: logger.info(s)
            )
            
            # 记录结束时间
            end_time = time.time()
            total_time = end_time - start_time
            
            total_images = sum(item.image_count for item in results)
            total_groups = sum(item.group_count for item in results)
            
            print(f"文件夹数量: {len(test_folders)}")
            print(f"总图片数量: {total_images}")
            print(f"总分组数量: {total_groups}")
            print(f"总耗时: {total_time:.2f} 秒")
            print(f"平均每张图片耗时: {total_time / total_images:.4f} 秒")
            print()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

# 测试不同规模的性能
def test_scalability():
    """测试不同规模的性能"""
    print("=== 测试不同规模的性能 ===")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for size in [10, 50]:  # 减小测试规模以加快速度
                # 创建测试文件夹和图片
                test_folder = Path(tmpdir) / f"test_album_{size}"
                create_test_images(test_folder, size)
                
                # 检查图片是否创建成功
                images = list_images(test_folder)
                print(f"图片数量: {len(images)}")
                
                if not images:
                    print("错误: 没有找到图片")
                    continue
                
                # 记录开始时间
                start_time = time.time()
                
                # 执行拆分
                logger.info(f"开始执行拆分，图片数量: {size}")
                result = extract_and_split(test_folder, on_status=lambda s: logger.info(s))
                
                # 记录结束时间
                end_time = time.time()
                total_time = end_time - start_time
                
                print(f"图片数量: {size}")
                print(f"分组数量: {len(result.groups)}")
                print(f"总耗时: {total_time:.2f} 秒")
                print(f"平均每张图片耗时: {total_time / size:.4f} 秒")
                print()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("开始测试拆分功能性能...")
    test_single_folder_performance()
    test_batch_folder_performance()
    test_scalability()
    print("测试完成!")
