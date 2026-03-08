import re

with open("tests/test_telegram_auth.py", "r") as f:
    code = f.read()

# Mock out dependencies explicitly inside the test
mock_code = """
        try:
            import src.reporting.telegram_bot as tb
            # Brutal mock container
            tb.container = MagicMock()
            tb.container.get.return_value = MagicMock()

            from src.reporting.telegram_bot import TelegramBot
"""

code = code.replace("        try:\n            from src.reporting.telegram_bot import TelegramBot", mock_code)

with open("tests/test_telegram_auth.py", "w") as f:
    f.write(code)

print("Patched test_telegram_auth.py again.")
