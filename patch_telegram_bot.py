import re

with open("src/reporting/telegram_bot.py", "r") as f:
    code = f.read()

replacement = """        # Extensions
        try:
            from src.system.container import container
            self.smart_money = container.get('smart_money')
        except Exception:
            self.smart_money = None"""

code = re.sub(r'        # Extensions\n        self\.smart_money = container\.get\(\'smart_money\'\)', replacement, code, flags=re.DOTALL)

with open("src/reporting/telegram_bot.py", "w") as f:
    f.write(code)

print("Patched telegram_bot.py container import properly.")
