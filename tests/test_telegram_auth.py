from unittest.mock import patch, MagicMock, AsyncMock
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
            import src.ui.telegram_mini_app as tb
            tb.container = MagicMock()
            tb.container.get.return_value = MagicMock()

            from src.ui.telegram_mini_app import TelegramApp as TelegramBot

            bot = TelegramBot()
            sentinel_mock = MagicMock()
            bot.set_sentinel(sentinel_mock)

            cases = [
                (111111, True, "Allowed User 1"),
                (222222, True, "Allowed User 2"),
                (333333, False, "Legacy Admin"),
                (444444, False, "Unauthorized User"),
            ]

            async def run_cases():
                for chat_id, expected, name in cases:
                    sentinel_mock.reset_mock()

                    mock_update = MagicMock()
                    mock_update.effective_user.id = chat_id
                    mock_update.message.reply_text = AsyncMock()

                    await bot._cmd_shutdown(mock_update, None)

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
