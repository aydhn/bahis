import re

with open("tests/test_physics_stage.py", "r") as f:
    code = f.read()

if "import pytest" not in code:
    code = "import pytest\n" + code

code = code.replace("async def test_physics_stage():", "@pytest.mark.asyncio\nasync def test_physics_stage():")

with open("tests/test_physics_stage.py", "w") as f:
    f.write(code)

print("Patched test_physics_stage.py")
