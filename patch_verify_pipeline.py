import re

with open("tests/verify_pipeline.py", "r") as f:
    code = f.read()

# Make it a proper test
code = "import pytest\n" + code
code = code.replace("async def main():", "@pytest.mark.asyncio\nasync def test_main():")
code = code.replace("if __name__ == \"__main__\":\n    asyncio.run(main())", "")

with open("tests/verify_pipeline.py", "w") as f:
    f.write(code)

print("Patched verify_pipeline.py")
