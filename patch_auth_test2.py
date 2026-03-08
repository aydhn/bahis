import re

with open("tests/test_telegram_auth.py", "r") as f:
    code = f.read()

# Make the container explicitly available in the test namespace if mocked
new_code = """
from unittest.mock import patch, MagicMock
import sys
import unittest
import asyncio
import os

container_mock = MagicMock()
container_module = MagicMock(container=container_mock)
sys.modules['src.system.container'] = container_module

class TestTelegramAuth(unittest.TestCase):
"""

code = re.sub(r'from unittest\.mock.*?class TestTelegramAuth\(unittest\.TestCase\):', new_code, code, flags=re.DOTALL)

with open("tests/test_telegram_auth.py", "w") as f:
    f.write(code)

print("Patched test_telegram_auth.py again")
