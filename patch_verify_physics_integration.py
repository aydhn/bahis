import re

with open("tests/verify_physics_integration.py", "r") as f:
    code = f.read()

replacement = """        class DummyBot:
            def __init__(self):
                self.enabled = False
            async def start(self): pass
            def set_context(self, ctx): pass
            def set_performance_report(self, r): pass
            async def send_bet_signal(self, b): pass
            async def send_risk_alert(self, t, m): pass
            async def send_godmode_report(self, ctx): return True

        reporter = ReportingStage(bot_instance=DummyBot())"""

code = code.replace("""        # Use a mock bot to avoid real telegram connection
        reporter = ReportingStage(bot_instance=MagicMock())""", replacement)

with open("tests/verify_physics_integration.py", "w") as f:
    f.write(code)
