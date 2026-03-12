"""简单测试脚本"""

import random


def generate_human_track(distance: int) -> list[tuple[int, int, int]]:
    """生成模拟人类的拖拽轨迹。"""
    if distance <= 0:
        return []

    track: list[tuple[int, int, int]] = []
    traveled = 0

    track.append((0, 0, random.randint(80, 200)))

    phase1 = distance * 0.15
    phase2 = distance * 0.70

    while traveled < distance:
        remaining = distance - traveled
        if traveled < phase1:
            step_f = random.uniform(1, 3)
            dt = random.randint(18, 35)
        elif traveled < phase2:
            step_f = random.uniform(5, 12)
            dt = random.randint(6, 18)
        else:
            ratio = remaining / distance
            step_f = max(1.0, random.uniform(1, 3) * ratio * 4)
            dt = random.randint(20, 45)

        dx = max(1, min(round(step_f), remaining))
        traveled += dx

        y_range = 3 if traveled < phase2 else 1
        dy = random.randint(-y_range, y_range)
        track.append((dx, dy, dt))

    track.append((0, 0, random.randint(30, 80)))

    return track


def test():
    print("=== 测试轨迹生成 ===")
    
    test_distances = [100, 200, 246, 300]
    
    for distance in test_distances:
        print(f"\n测试距离: {distance}px")
        
        random.seed(42)
        
        track = generate_human_track(distance)
        print(f"轨迹步数: {len(track)}")
        
        total_dx = sum(dx for dx, dy, dt in track)
        print(f"总 X 位移: {total_dx}px")
        
        if total_dx != distance:
            print(f"❌ 警告: 总位移 {total_dx} != 目标距离 {distance}")
        else:
            print(f"✓ 总位移正确")


if __name__ == "__main__":
    test()
