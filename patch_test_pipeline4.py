import re

with open("tests/test_pipeline_stages.py", "r") as f:
    code = f.read()

# Add from unittest.mock import patch
code = code.replace("from unittest.mock import MagicMock", "from unittest.mock import MagicMock, patch")

with open("tests/test_pipeline_stages.py", "w") as f:
    f.write(code)
