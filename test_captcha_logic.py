"""测试滑块验证逻辑"""

import random
from yupoo_scraper.captcha_solver import generate_human_track


def test_track_generation():
    """测试轨迹生成"""
    print("=== 测试轨迹生成 ===")
    
    # 测试不同距离
    test_distances = [100, 200, 246, 300]
    
    for distance in test_distances:
        print(f"\n测试距离: {distance}px")
        
        # 设置随机种子保证可重现
        random.seed(42)
        
        track = generate_human_track(distance)
        print(f"轨迹步数: {len(track)}")
        
        # 计算总位移
        total_dx = sum(dx for dx, dy, dt in track)
        print(f"总 X 位移: {total_dx}px")
        
        if total_dx != distance:
            print(f"❌ 警告: 总位移 {total_dx} != 目标距离 {distance}")
        else:
            print(f"✓ 总位移正确")
        
        # 打印前几步和最后几步
        print("\n前 5 步:")
        for i, (dx, dy, dt) in enumerate(track[:5]):
            print(f"  步 {i+1}: dx={dx}, dy={dy}, dt={dt}ms")
        
        print("\n后 5 步:")
        for i, (dx, dy, dt) in enumerate(track[-5:]):
            print(f"  步 {len(track)-5+i+1}: dx={dx}, dy={dy}, dt={dt}ms")


if __name__ == "__main__":
    test_track_generation()
