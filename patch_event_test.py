import re

with open("tests/test_event_flow.py", "r") as f:
    code = f.read()

code = "import pytest\n" + code
code = code.replace("async def test_flow():", "@pytest.mark.asyncio\nasync def test_flow():")

with open("tests/test_event_flow.py", "w") as f:
    f.write(code)

print("Patched test_event_flow.py")
