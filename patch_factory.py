import re

file_path = "src/ingestion/async_data_factory.py"
with open(file_path, "r") as f:
    content = f.read()

# 1. Add imports at the top
if "import time" not in content:
    content = content.replace("import asyncio", "import asyncio\nimport time\nfrom src.extensions.smart_money import SmartMoneyDetector")

# 2. Add to __init__
init_pattern = r"(self\._browser = None\n\s+logger\.debug\(\"DataFactory başlatıldı\.\"\))"
if "self.smart_money = SmartMoneyDetector()" not in content:
    content = re.sub(init_pattern, r"\1\n        self.smart_money = SmartMoneyDetector()", content)

# 3. Add to _normalize_and_store
store_pattern = r"(odds = self\._extract_odds\(item\)\n\s+normalized\.update\(odds\))"
steam_code = """\\1

            if "home_odds" in odds:
                steam_signal = self.smart_money.detect_steam(match_id, odds["home_odds"], time.time())
                if steam_signal:
                    logger.info(f"Steam Move Detected for {match_id}: {steam_signal}")"""

if "self.smart_money.detect_steam" not in content:
    content = re.sub(store_pattern, steam_code, content)

with open(file_path, "w") as f:
    f.write(content)

print("async_data_factory.py patched successfully.")
