from unittest.mock import patch
import sys
from unittest.mock import MagicMock
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.stats"] = MagicMock()
sys.modules["scipy.optimize"] = MagicMock()
sys.modules["scipy.special"] = MagicMock()
sys.modules["scipy.integrate"] = MagicMock()
import unittest
import asyncio
import os

# Ensure src is in path
sys.path.append(os.getcwd())

class TestTelegramAuth(unittest.TestCase):
    def setUp(self):
        # Patch heavy dependencies
        self.patchers = []
        for mod in ['polars', 'numpy', 'matplotlib', 'matplotlib.pyplot',
                    'src.reporting.visualizer', 'src.ingestion.voice_interrogator',
                    'src.pipeline.context']:
            p = patch.dict(sys.modules, {mod: MagicMock()})
            p.start()
            self.patchers.append(p)

    def tearDown(self):
        for p in self.patchers:
            p.stop()

    @patch.dict(os.environ, {"TELEGRAM_TOKEN": "123:dummy", "TELEGRAM_CHAT_ID": "333333"})
    def test_authorization(self):
        # Import config and modify settings
        from src.system.config import settings
        # Save original value
        original_allowed = settings.TELEGRAM_ALLOWED_USERS
        settings.TELEGRAM_ALLOWED_USERS = "111111,222222"

        try:
            from src.reporting.telegram_bot import TelegramBot

            bot = TelegramBot()
            sentinel_mock = MagicMock()
            bot.set_sentinel(sentinel_mock)

            cases = [
                (111111, True, "Allowed User 1"),
                (222222, True, "Allowed User 2"),
                (333333, True, "Legacy Admin"),
                (444444, False, "Unauthorized User"),
            ]

            async def run_cases():
                for chat_id, expected, name in cases:
                    sentinel_mock.reset_mock()
                    update = {
                        "update_id": 1,
                        "message": {
                            "chat": {"id": chat_id},
                            "text": "/shutdown"
                        }
                    }

                    await bot._handle_update(update)

                    if expected:
                        self.assertTrue(sentinel_mock.shutdown.called, f"{name} failed to authorize")
                    else:
                        self.assertFalse(sentinel_mock.shutdown.called, f"{name} should be blocked")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_cases())
            loop.close()

        finally:
            settings.TELEGRAM_ALLOWED_USERS = original_allowed

if __name__ == '__main__':
    unittest.main()
