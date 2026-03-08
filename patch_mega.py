import re

with open("tests/test_mega_refactor.py", "r") as f:
    code = f.read()

code = code.replace("from tests.conftest import MockDB", "")

with open("tests/test_mega_refactor.py", "w") as f:
    f.write(code)
