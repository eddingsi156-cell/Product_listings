"""测试 open_clip 库的基本功能"""

import sys
import time

def test_open_clip():
    """测试 open_clip 库的基本功能"""
    print("=== 测试 open_clip 库 ===")
    
    try:
        # 测试导入
        print("导入 open_clip...")
        import open_clip
        print("导入成功")
        
        # 测试模型创建
        print("创建模型...")
        start_time = time.time()
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        end_time = time.time()
        print(f"模型创建成功，耗时: {end_time - start_time:.2f} 秒")
        print(f"模型类型: {type(model)}")
        
        # 测试模型移动到 CPU
        print("移动模型到 CPU...")
        model = model.to("cpu").eval()
        print("移动成功")
        
        # 测试模型信息
        print("模型信息:")
        print(f"  模型名称: ViT-B-32")
        print(f"  预训练权重: openai")
        print(f"  模型状态: 已加载")
        
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试 open_clip 库...")
    
    # 测试 open_clip
    test_ok = test_open_clip()
    print(f"\nopen_clip 测试: {'通过' if test_ok else '失败'}")
    
    print("\n测试完成!")
