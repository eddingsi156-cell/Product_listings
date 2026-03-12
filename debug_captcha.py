"""滑块验证码调试脚本

连接到本地 Chrome CDP 端口，手动测试滑块验证逻辑。
需要先启动 Chrome 浏览器并开启远程调试端口：
chrome.exe --remote-debugging-port=9222
"""

import asyncio
import logging
from pathlib import Path

from playwright.async_api import async_playwright

from yupoo_scraper import config
from yupoo_scraper.uploader import WeidianUploader

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def debug_captcha():
    """调试滑块验证码"""
    logger.info("=== 滑块验证码调试模式 ===")
    
    async with async_playwright() as p:
        # 连接到已打开的 Chrome 浏览器
        logger.info(f"正在连接到 Chrome CDP: {config.WEIDIAN_CDP_URL}")
        browser = await p.chromium.connect_over_cdp(config.WEIDIAN_CDP_URL)
        
        # 获取第一个页面
        context = browser.contexts[0]
        page = context.pages[0]
        logger.info(f"已连接到页面: {page.url}")
        
        # 创建上传器实例
        uploader = WeidianUploader()
        
        # 简单的步骤回调
        def step(msg: str):
            logger.info(f"[STEP] {msg}")
        
        # 等待用户确认
        logger.info("\n=== 请在浏览器中导航到会触发滑块验证码的页面 ===")
        logger.info("准备好后按回车键继续...")
        input()
        
        # 尝试处理验证码
        logger.info("开始处理验证码...")
        success = await uploader._handle_captcha(page, step)
        
        if success:
            logger.info("✓ 验证码处理成功！")
        else:
            logger.error("✗ 验证码处理失败")
        
        # 保持浏览器连接，方便调试
        logger.info("\n调试完成，浏览器保持连接。按 Ctrl+C 退出...")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("退出调试模式")


if __name__ == "__main__":
    asyncio.run(debug_captcha())
