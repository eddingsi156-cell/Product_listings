"""滑块验证码识别 — 第三方打码平台集成

支持平台:
  - ttshitu (图鉴, ttshitu.com) — 国内推荐，便宜（~2分/次）
  - twocaptcha (2Captcha, 2captcha.com) — 国际平台

用法:
  solver = create_solver("ttshitu", username="xxx", password="xxx")
  gap_x = await solver.recognize_gap(screenshot_bytes)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random

import aiohttp

logger = logging.getLogger(__name__)


# ── 滑块拖拽模拟 ─────────────────────────────────────────────────


def generate_human_track(distance: int) -> list[tuple[int, int, int]]:
    """生成模拟人类的拖拽轨迹。

    Returns:
        [(dx, dy, duration_ms), ...] 相对位移序列。
        dx 之和精确等于 distance（过冲回调除外，其净额为 0）。
    """
    if distance <= 0:
        return []

    track: list[tuple[int, int, int]] = []
    traveled = 0  # 整数累加，保证精确

    # 按下后短暂停顿（模拟反应时间）
    track.append((0, 0, random.randint(80, 200)))

    # 三阶段：慢启动 → 快速滑动 → 精确减速
    phase1 = distance * 0.15  # 慢启动
    phase2 = distance * 0.70  # 快速阶段结束点

    while traveled < distance:
        remaining = distance - traveled
        if traveled < phase1:
            # 慢启动：小步、较长间隔
            step_f = random.uniform(1, 3)
            dt = random.randint(18, 35)
        elif traveled < phase2:
            # 快速阶段：大步、短间隔
            step_f = random.uniform(5, 12)
            dt = random.randint(6, 18)
        else:
            # 精确减速：越接近终点步长越小
            ratio = remaining / distance
            step_f = max(1.0, random.uniform(1, 3) * ratio * 4)
            dt = random.randint(20, 45)

        dx = max(1, min(round(step_f), remaining))
        traveled += dx

        # 轻微 Y 轴抖动（快速阶段抖动更大）
        y_range = 3 if traveled < phase2 else 1
        dy = random.randint(-y_range, y_range)
        track.append((dx, dy, dt))

    # 到达后短暂停顿（模拟确认）
    track.append((0, 0, random.randint(30, 80)))

    # 过冲 + 回调（概率 40%，更拟人）
    if random.random() < 0.4:
        overshoot = random.randint(3, 8)
        track.append((overshoot, random.randint(-1, 1), random.randint(20, 50)))
        track.append((-overshoot, random.randint(-1, 1), random.randint(60, 120)))

    return track


# ── 打码平台基类 ──────────────────────────────────────────────────


class CaptchaSolver:
    """打码平台抽象基类。

    在 async with 块中使用以复用 HTTP 连接，或直接调用（每次新建连接）。
    """

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None
        return False

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取复用的 session，或创建临时 session。"""
        if self._session and not self._session.closed:
            return self._session
        # 先关闭旧 session（如果存在但已关闭），防止泄漏
        if self._session is not None:
            self._session = None
        self._session = aiohttp.ClientSession()
        return self._session

    async def recognize_gap(self, image_bytes: bytes) -> int:
        """识别滑块缺口的 X 坐标。

        Args:
            image_bytes: 验证码截图的 PNG/JPEG 字节。

        Returns:
            缺口的 X 像素坐标（相对于图片左侧）。
        """
        raise NotImplementedError



# ── TTShitu (图鉴) ────────────────────────────────────────────────


class TTShituSolver(CaptchaSolver):
    """图鉴打码平台 (ttshitu.com)。

    注册后获取 username/password。
    滑块验证码 typeId = 33（滑块缺口识别，返回缺口 X 坐标）。
    """

    API_URL = "https://api.ttshitu.com/predict"

    def __init__(self, username: str, password: str, type_id: int = 33):
        super().__init__()
        self._username = username
        self._password = password
        self._type_id = type_id  # 33=滑块缺口识别

    async def recognize_gap(self, image_bytes: bytes) -> int:
        b64 = base64.b64encode(image_bytes).decode()

        payload = {
            "username": self._username,
            "password": self._password,
            "typeid": self._type_id,
            "image": b64,
        }

        session = await self._get_session()
        async with session.post(
            self.API_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json()

        if not data.get("success"):
            msg = data.get("message", "未知错误")
            # 区分不可重试错误（余额不足、账号问题）和临时错误
            fatal_keywords = ("余额", "用户名", "密码", "账号", "封禁", "冻结")
            if any(kw in msg for kw in fatal_keywords):
                raise ValueError(f"图鉴账号错误（不可重试）: {msg}")
            raise RuntimeError(f"图鉴识别失败: {msg}")

        result = data["data"]["result"]
        logger.info("图鉴返回结果: %s", result)

        # 返回格式可能是 "x,y" 或纯数字
        if "," in str(result):
            x_str = str(result).split(",")[0]
            return int(float(x_str))
        return int(float(result))


# ── 2Captcha ──────────────────────────────────────────────────────


class TwoCaptchaSolver(CaptchaSolver):
    """2Captcha 打码平台 (2captcha.com)。

    使用 type=slider 提交滑块验证码截图，返回滑块缺口的 X 坐标。
    """

    # 2Captcha JSON API v2
    SUBMIT_URL = "https://api.2captcha.com/createTask"
    RESULT_URL = "https://api.2captcha.com/getTaskResult"

    def __init__(self, api_key: str):
        super().__init__()
        self._api_key = api_key

    async def recognize_gap(self, image_bytes: bytes) -> int:
        b64 = base64.b64encode(image_bytes).decode()

        # 使用 SliderCaptcha 任务类型
        payload = {
            "clientKey": self._api_key,
            "task": {
                "type": "SliderCaptchaTask",
                "body": b64,
            },
        }

        session = await self._get_session()

        # 提交
        async with session.post(
            self.SUBMIT_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json()

        if data.get("errorId", 0) != 0:
            raise RuntimeError(
                f"2Captcha 提交失败: {data.get('errorDescription', data.get('errorCode', '未知错误'))}"
            )

        task_id = data["taskId"]
        logger.info("2Captcha 任务已提交: %s", task_id)

        # 官方建议首次等 5 秒再轮询，之后每 2 秒（最多 ~65 秒）
        await asyncio.sleep(5)
        for _ in range(30):
            await asyncio.sleep(2)

            async with session.post(
                self.RESULT_URL,
                json={"clientKey": self._api_key, "taskId": task_id},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                result = await resp.json()

            if result.get("status") == "ready":
                solution = result.get("solution", {})
                logger.info("2Captcha 返回结果: %s", solution)
                # SliderCaptcha 返回 slideDistance 或 x 坐标
                if "slideDistance" in solution:
                    return int(float(solution["slideDistance"]))
                if "x" in solution:
                    return int(float(solution["x"]))
                raise RuntimeError(f"2Captcha 未返回坐标: {solution}")

            if result.get("status") != "processing":
                raise RuntimeError(
                    f"2Captcha 识别失败: {result.get('errorDescription', result.get('errorCode'))}"
                )

        raise RuntimeError("2Captcha 识别超时")


# ── 工厂函数 ──────────────────────────────────────────────────────


def create_solver(
    provider: str,
    *,
    username: str = "",
    password: str = "",
    api_key: str = "",
) -> CaptchaSolver:
    """创建打码平台实例。

    Args:
        provider: "ttshitu" 或 "twocaptcha"。
        username: 图鉴用户名。
        password: 图鉴密码。
        api_key: 2Captcha API Key。

    Returns:
        CaptchaSolver 实例。
    """
    provider = provider.lower().strip()
    if provider == "ttshitu":
        if not username or not password:
            raise ValueError("图鉴平台需要 username 和 password")
        return TTShituSolver(username, password)
    elif provider in ("twocaptcha", "2captcha"):
        if not api_key:
            raise ValueError("2Captcha 需要 api_key")
        return TwoCaptchaSolver(api_key)
    else:
        raise ValueError(f"不支持的打码平台: {provider!r}，可选: ttshitu, twocaptcha")
