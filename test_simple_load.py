"""简单测试特征提取器的 load_model 方法"""

import sys
import time
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

print("开始测试...")

# 直接导入特征提取器
print("导入特征提取器...")
from yupoo_scraper.ml.feature_extractor import get_extractor

# 获取提取器实例
print("获取提取器实例...")
extractor = get_extractor()
print(f"提取器实例: {extractor}")
print(f"设备: {extractor.device}")
print(f"模型是否已加载: {extractor.loaded}")

# 加载模型
print("开始加载模型...")
start_time = time.time()

try:
    extractor.load_model(
        on_progress=lambda msg: print(f"模型加载进度: {msg}")
    )
    end_time = time.time()
    print(f"模型加载完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"模型加载状态: {extractor.loaded}")
    print("测试成功!")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
