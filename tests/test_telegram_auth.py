from unittest.mock import patch, MagicMock
import sys
import unittest
import asyncio
import os

import numpy as np
import scipy

container_mock = MagicMock()
container_module = MagicMock(container=container_mock)
sys.modules['src.system.container'] = container_module

class TestTelegramAuth(unittest.TestCase):
    @patch.dict(os.environ, {"TELEGRAM_TOKEN": "123:dummy", "TELEGRAM_CHAT_ID": "333333"})
    def test_authorization(self):
        from src.system.config import settings
        original_allowed = settings.TELEGRAM_ALLOWED_USERS
        settings.TELEGRAM_ALLOWED_USERS = "111111,222222"


        try:
            import src.reporting.telegram_bot as tb
            # Brutal mock container
            tb.container = MagicMock()
            tb.container.get.return_value = MagicMock()

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
