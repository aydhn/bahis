import re

with open("src/pipeline/stages/reporting.py", "r") as f:
    code = f.read()

# Make TelegramBot instantiate gracefully
replacement = """    def __init__(self, bot_instance: Optional[TelegramBot] = None):
        super().__init__("reporting")
        self.bot = bot_instance
        try:
            self.ceo_dashboard = CEODashboard()
        except Exception:
            self.ceo_dashboard = None

        if self.bot is None:
            try:
                self.bot = TelegramBot()
            except Exception:
                self.bot = None

        # Botu başlat (Arka planda polling)
        if self.bot and getattr(self.bot, 'enabled', False):
            asyncio.create_task(self.bot.start())"""

code = re.sub(r'    def __init__\(self, bot_instance: Optional\[TelegramBot\] = None\):.*?if bot_instance is None and self\.bot\.enabled:\n            asyncio\.create_task\(self\.bot\.start\(\)\)', replacement, code, flags=re.DOTALL)

with open("src/pipeline/stages/reporting.py", "w") as f:
    f.write(code)

print("Patched reporting.py")
