"""交互式滑块验证码调试

手动测试不同的滑动距离，找出正确的值。
"""

import asyncio
import logging
import random
from playwright.async_api import async_playwright

from yupoo_scraper import config


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_human_track(distance: int) -> list[tuple[int, int, int]]:
    """生成模拟人类的拖拽轨迹（简化版，无过冲）"""
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


async def perform_slide(page, start_x: float, start_y: float, distance: int) -> None:
    """执行滑块拖拽"""
    logger.info(f"开始滑动: 距离={distance}px")
    
    await page.mouse.move(start_x, start_y)
    await asyncio.sleep(random.uniform(0.1, 0.3))
    await page.mouse.down()
    await asyncio.sleep(random.uniform(0.05, 0.15))

    track = generate_human_track(distance)
    cx, cy = start_x, start_y
    for dx, dy, dt in track:
        if dx == 0 and dy == 0:
            await asyncio.sleep(dt / 1000)
            continue
        cx += dx
        cy += dy
        await page.mouse.move(cx, cy)
        await asyncio.sleep(dt / 1000)

    await asyncio.sleep(random.uniform(0.05, 0.2))
    await page.mouse.up()
    logger.info("滑动完成")


async def main():
    """主函数"""
    print("=== 交互式滑块验证码调试 ===")
    print("请确保 Chrome 已启动并开启远程调试:")
    print("  chrome.exe --remote-debugging-port=9222")
    print()
    
    async with async_playwright() as p:
        try:
            logger.info(f"正在连接到 Chrome CDP: {config.WEIDIAN_CDP_URL}")
            browser = await p.chromium.connect_over_cdp(config.WEIDIAN_CDP_URL)
        except Exception as e:
            logger.error(f"连接失败: {e}")
            logger.error("请确保 Chrome 已启动并开启 --remote-debugging-port=9222")
            return

        context = browser.contexts[0]
        page = context.pages[0]
        logger.info(f"已连接到页面: {page.url}")

        print("\n=== 准备工作 ===")
        print("1. 请在浏览器中导航到会触发滑块验证码的页面")
        print("2. 触发滑块验证码")
        print("3. 准备好后按回车键继续")
        input()

        while True:
            print("\n" + "="*50)
            print("请选择操作:")
            print("  1. 查找滑块并获取信息")
            print("  2. 尝试特定的滑动距离")
            print("  3. 测试一系列距离")
            print("  q. 退出")
            
            choice = input("\n请输入选项: ").strip().lower()
            
            if choice == 'q':
                break
            
            if choice == '1':
                await get_slider_info(page)
            elif choice == '2':
                await test_specific_distance(page)
            elif choice == '3':
                await test_multiple_distances(page)
            else:
                print("无效选项")


async def get_slider_info(page):
    """获取滑块和验证码信息"""
    print("\n--- 获取滑块信息 ---")
    
    try:
        frame = page.frame_locator("#tcaptcha_iframe")
        
        # 查找滑块
        slider_selectors = [
            "#tcaptcha_drag_button",
            ".tc-drag-thumb",
            "#tcaptcha_drag_thumb",
        ]
        
        slider = None
        for sel in slider_selectors:
            try:
                loc = frame.locator(sel).first
                if await loc.is_visible(timeout=2000):
                    slider = loc
                    print(f"找到滑块: {sel}")
                    break
            except Exception:
                continue
        
        if not slider:
            print("未找到滑块")
            return
        
        slider_box = await slider.bounding_box()
        print(f"滑块位置: {slider_box}")
        
        # 查找背景图
        bg_selectors = ["#slideBg", "img.tc-bg-img"]
        bg_img = None
        for sel in bg_selectors:
            try:
                loc = frame.locator(sel).first
                if await loc.is_visible(timeout=2000):
                    bg_img = loc
                    print(f"找到背景图: {sel}")
                    break
            except Exception:
                continue
        
        if bg_img:
            bg_box = await bg_img.bounding_box()
            print(f"背景图位置: {bg_box}")
        
        # 尝试获取拼图块信息
        try:
            tc_frame = None
            for f in page.frames:
                if "cap_union_new_show" in (f.url or ""):
                    tc_frame = f
                    break
            
            if tc_frame:
                piece_info = await tc_frame.evaluate("""() => {
                    const el = document.getElementById('slideBlock');
                    if (!el) return null;
                    return {
                        left: parseFloat(el.style.left),
                        width: parseFloat(el.style.width) || el.offsetWidth,
                        top: parseFloat(el.style.top),
                        height: parseFloat(el.style.height) || el.offsetHeight,
                    };
                }""")
                print(f"拼图块信息: {piece_info}")
        except Exception as e:
            print(f"获取拼图块信息失败: {e}")
        
    except Exception as e:
        print(f"错误: {e}")


async def test_specific_distance(page):
    """测试特定的滑动距离"""
    print("\n--- 测试特定滑动距离 ---")
    
    try:
        frame = page.frame_locator("#tcaptcha_iframe")
        
        # 查找滑块
        slider_selectors = [
            "#tcaptcha_drag_button",
            ".tc-drag-thumb",
            "#tcaptcha_drag_thumb",
        ]
        
        slider = None
        for sel in slider_selectors:
            try:
                loc = frame.locator(sel).first
                if await loc.is_visible(timeout=2000):
                    slider = loc
                    break
            except Exception:
                continue
        
        if not slider:
            print("未找到滑块")
            return
        
        slider_box = await slider.bounding_box()
        start_x = slider_box["x"] + slider_box["width"] / 2
        start_y = slider_box["y"] + slider_box["height"] / 2
        
        distance_str = input("请输入要测试的滑动距离 (像素): ").strip()
        try:
            distance = int(distance_str)
        except ValueError:
            print("无效的距离")
            return
        
        confirm = input(f"确定要滑动 {distance}px 吗? (y/n): ").strip().lower()
        if confirm != 'y':
            return
        
        await perform_slide(page, start_x, start_y, distance)
        
        print("滑动完成，请观察验证码结果")
        print("如果成功，请记录这个距离值！")
        
    except Exception as e:
        print(f"错误: {e}")


async def test_multiple_distances(page):
    """测试一系列距离"""
    print("\n--- 测试一系列距离 ---")
    
    try:
        frame = page.frame_locator("#tcaptcha_iframe")
        
        # 查找滑块
        slider_selectors = [
            "#tcaptcha_drag_button",
            ".tc-drag-thumb",
            "#tcaptcha_drag_thumb",
        ]
        
        slider = None
        for sel in slider_selectors:
            try:
                loc = frame.locator(sel).first
                if await loc.is_visible(timeout=2000):
                    slider = loc
                    break
            except Exception:
                continue
        
        if not slider:
            print("未找到滑块")
            return
        
        slider_box = await slider.bounding_box()
        start_x = slider_box["x"] + slider_box["width"] / 2
        start_y = slider_box["y"] + slider_box["height"] / 2
        
        base_str = input("请输入基础距离 (图鉴返回的 gap_x): ").strip()
        try:
            base = int(base_str)
        except ValueError:
            print("无效的距离")
            return
        
        print(f"将测试以下距离 (基于 {base}):")
        offsets = [-30, -20, -10, -5, 0, 5, 10, 20, 30]
        test_distances = [base + offset for offset in offsets]
        print(f"  {test_distances}")
        
        confirm = input("确定要继续吗? 每次测试后需要手动刷新验证码 (y/n): ").strip().lower()
        if confirm != 'y':
            return
        
        for distance in test_distances:
            print(f"\n测试距离: {distance}px")
            
            # 重新获取滑块位置（可能刷新了）
            try:
                slider_box = await slider.bounding_box(timeout=2000)
                start_x = slider_box["x"] + slider_box["width"] / 2
                start_y = slider_box["y"] + slider_box["height"] / 2
            except Exception:
                print("未找到滑块，请先刷新验证码，然后按回车键继续...")
                input()
                # 重新查找滑块
                slider = None
                for sel in slider_selectors:
                    try:
                        loc = frame.locator(sel).first
                        if await loc.is_visible(timeout=2000):
                            slider = loc
                            break
                    except Exception:
                        continue
                if not slider:
                    print("还是找不到滑块，跳过此距离")
                    continue
                slider_box = await slider.bounding_box()
                start_x = slider_box["x"] + slider_box["width"] / 2
                start_y = slider_box["y"] + slider_box["height"] / 2
            
            await perform_slide(page, start_x, start_y, distance)
            
            print(f"已滑动 {distance}px")
            print("请观察结果，如果成功请记录这个距离！")
            print("刷新验证码后按回车键继续下一个测试...")
            input()
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())
