"""
stealth_browser.py – Anti-Detection Headless Browser.

undetected-chromedriver veya Playwright stealth plugin ile
bot tespitinden kaçınır. User-Agent havuzu + fingerprint rotasyonu.
"""
from __future__ import annotations

import asyncio
import random
import shutil

from loguru import logger

# ── User-Agent Havuzu (güncel, gerçek tarayıcılar) ──
_UA_POOL: list[str] = [
    # Chrome 120-124 (Windows)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Chrome (Mac)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    # Firefox
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0",
    # Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    # Safari
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

# ── Viewport Havuzu ──
_VIEWPORTS = [
    {"width": 1920, "height": 1080},
    {"width": 1366, "height": 768},
    {"width": 1536, "height": 864},
    {"width": 1440, "height": 900},
    {"width": 1280, "height": 720},
    {"width": 2560, "height": 1440},
]

# ── Locale Havuzu ──
_LOCALES = ["tr-TR", "en-US", "en-GB", "de-DE"]
_TIMEZONES = ["Europe/Istanbul", "Europe/Berlin", "Europe/London", "America/New_York"]


def random_ua() -> str:
    """Havuzdan rastgele User-Agent seç."""
    try:
        from fake_useragent import UserAgent
        return UserAgent().random
    except Exception:
        return random.choice(_UA_POOL)


def random_fingerprint() -> dict:
    """Rastgele tarayıcı parmak izi üret."""
    return {
        "user_agent": random_ua(),
        "viewport": random.choice(_VIEWPORTS),
        "locale": random.choice(_LOCALES),
        "timezone": random.choice(_TIMEZONES),
        "color_depth": random.choice([24, 32]),
        "device_scale_factor": random.choice([1, 1.25, 1.5, 2]),
    }


class StealthBrowser:
    """Anti-detection headless browser.

    Öncelik sırası:
    1. undetected-chromedriver (en iyi anti-detection)
    2. Playwright + stealth plugin
    3. Standart Playwright

    Her 10 istekte fingerprint rotasyonu yapar.
    """

    def __init__(self, headless: bool = True, engine: str = "auto"):
        self._headless = headless
        self._engine = engine
        self._browser = None
        self._context = None
        self._page = None
        self._request_count = 0
        self._fingerprint = random_fingerprint()
        self._active_engine: str = ""
        logger.debug(f"StealthBrowser başlatıldı (engine={engine}).")

    async def check_compatibility(self) -> bool:
        """Verifies if undetected-chromedriver and Chrome are compatible."""
        try:
            import importlib.util
            if importlib.util.find_spec("undetected_chromedriver") is None:
                raise ImportError("No module named \'undetected_chromedriver\'")
            # Basic check: verify Chrome binary exists
            chrome_path = shutil.which("google-chrome") or shutil.which("chrome") or shutil.which("chromium")
            if not chrome_path:
                logger.warning("Chrome binary not found in PATH.")
                return False

            # Additional version checks could be implemented here
            # by parsing `google-chrome --version` and uc.__version__
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.error(f"Compatibility check failed: {e}")
            return False

    async def start(self):
        """Tarayıcıyı başlatır – anti-detection sırasıyla dener."""
        if self._engine in ("auto", "undetected"):
            if await self.check_compatibility():
                if await self._try_undetected():
                    return
            else:
                logger.debug("Skipping undetected-chromedriver due to compatibility check.")

        if self._engine in ("auto", "playwright"):
            if await self._try_playwright_stealth():
                return
            if await self._try_playwright_standard():
                return

        logger.error("Hiçbir tarayıcı motoru başlatılamadı.")

    # ─── undetected-chromedriver ───
    async def _try_undetected(self) -> bool:
        try:
            import undetected_chromedriver as uc
            options = uc.ChromeOptions()
            if self._headless:
                options.add_argument("--headless=new")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument(f"--user-agent={self._fingerprint['user_agent']}")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument(f"--window-size={self._fingerprint['viewport']['width']},{self._fingerprint['viewport']['height']}")

            # Run in a separate thread because UC is blocking
            def launch():
                return uc.Chrome(options=options)

            self._browser = await asyncio.to_thread(launch)
            self._active_engine = "undetected-chromedriver"
            logger.success(f"StealthBrowser: {self._active_engine} başlatıldı.")
            return True
        except ImportError:
            logger.debug("undetected-chromedriver not installed.")
            return False
        except Exception as e:
            logger.debug(f"undetected-chromedriver başlatılamadı: {e}")
            return False

    # ─── Playwright + stealth ───
    async def _try_playwright_stealth(self) -> bool:
        try:
            from playwright.async_api import async_playwright
            from playwright_stealth import stealth_async
        except ImportError:
            return False

        try:
            pw = await async_playwright().start()
            self._browser = await pw.chromium.launch(headless=self._headless)
            fp = self._fingerprint
            self._context = await self._browser.new_context(
                user_agent=fp["user_agent"],
                viewport=fp["viewport"],
                locale=fp["locale"],
                timezone_id=fp["timezone"],
                color_scheme="light",
                device_scale_factor=fp["device_scale_factor"],
            )
            self._page = await self._context.new_page()
            await stealth_async(self._page)
            self._active_engine = "playwright-stealth"
            logger.success(f"StealthBrowser: {self._active_engine} başlatıldı.")
            return True
        except Exception as e:
            logger.debug(f"playwright-stealth başlatılamadı: {e}")
            return False

    # ─── Standart Playwright ───
    async def _try_playwright_standard(self) -> bool:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return False

        try:
            pw = await async_playwright().start()
            self._browser = await pw.chromium.launch(headless=self._headless)
            fp = self._fingerprint
            self._context = await self._browser.new_context(
                user_agent=fp["user_agent"],
                viewport=fp["viewport"],
                locale=fp["locale"],
                timezone_id=fp["timezone"],
            )
            self._page = await self._context.new_page()

            # Manuel stealth injection
            await self._page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                window.chrome = { runtime: {} };
                Object.defineProperty(navigator, 'languages', {get: () => ['tr-TR', 'tr', 'en']});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            """)
            self._active_engine = "playwright-standard"
            logger.success(f"StealthBrowser: {self._active_engine} başlatıldı.")
            return True
        except Exception as e:
            logger.debug(f"playwright başlatılamadı: {e}")
            return False

    # ─── Sayfa navigasyonu ───
    async def goto(self, url: str, wait_ms: int = 3000) -> str | None:
        """URL'ye git, HTML döndür. Her 10 istekte fingerprint döndür."""
        self._request_count += 1

        if self._request_count % 10 == 0:
            await self._rotate_fingerprint()

        # Rastgele gecikme (insan davranışı)
        await asyncio.sleep(random.uniform(1.0, 3.5))

        try:
            if self._active_engine == "undetected-chromedriver":
                await asyncio.to_thread(self._browser.get, url)
                await asyncio.sleep(wait_ms / 1000)
                return self._browser.page_source
            elif self._page:
                await self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await self._page.wait_for_timeout(wait_ms)
                return await self._page.content()
        except Exception as e:
            logger.warning(f"StealthBrowser goto hatası ({url[:50]}): {e}")
            return None

    # ─── Etkileşim (Page Actions) ───
    async def page_action(self, action_type: str, selector: str = "", value: str = ""):
        """Perform actions like click, type, scroll."""
        if not self._page and self._active_engine != "undetected-chromedriver":
            logger.warning("Browser not active or engine not supported for actions.")
            return

        try:
            if self._active_engine == "undetected-chromedriver":
                from selenium.webdriver.common.by import By

                # Simple mapping for selector (heuristic)
                by = By.CSS_SELECTOR
                if selector.startswith("//"):
                    by = By.XPATH
                elif selector.startswith("#"):
                    by = By.ID

                def _perform_selenium():
                    if action_type == "scroll":
                        self._browser.execute_script(f"window.scrollBy(0, {value or 300})")
                    else:
                        el = self._browser.find_element(by, selector)
                        if action_type == "click":
                            el.click()
                        elif action_type == "type":
                            el.send_keys(value)

                await asyncio.to_thread(_perform_selenium)

            elif self._page:
                if action_type == "click":
                    await self._page.click(selector)
                elif action_type == "type":
                    await self._page.fill(selector, value)
                elif action_type == "scroll":
                    # Scroll down by value pixels or defaults
                    await self._page.evaluate(f"window.scrollBy(0, {value or 300})")
                elif action_type == "wait":
                    await self._page.wait_for_selector(selector, timeout=5000)

        except Exception as e:
            logger.warning(f"Page action '{action_type}' failed on '{selector}': {e}")

    async def _rotate_fingerprint(self):
        """Fingerprint rotasyonu – yeni kimlik."""
        self._fingerprint = random_fingerprint()
        logger.debug(f"Fingerprint döndürüldü (istek #{self._request_count})")

        if self._context and self._active_engine.startswith("playwright"):
            try:
                await self._page.close()
                await self._context.close()
                fp = self._fingerprint
                self._context = await self._browser.new_context(
                    user_agent=fp["user_agent"],
                    viewport=fp["viewport"],
                    locale=fp["locale"],
                    timezone_id=fp["timezone"],
                )
                self._page = await self._context.new_page()
            except Exception as e:
                logger.debug(f"Fingerprint rotasyon hatası: {e}")

    async def close(self):
        try:
            if self._active_engine == "undetected-chromedriver" and self._browser:
                # Quit is blocking
                await asyncio.to_thread(self._browser.quit)
                self._browser = None
            elif self._browser:
                await self._browser.close()
                self._browser = None
        except Exception:
            pass

    @property
    def engine(self) -> str:
        return self._active_engine

    @property
    def request_count(self) -> int:
        return self._request_count
