import re

with open("tests/verify_pipeline.py", "r") as f:
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
            async def send_message(self, report): pass

        reporter = ReportingStage(bot_instance=DummyBot())

        # Mock CEO dashboard to return async strings to avoid MagicMock await error
        async def mock_enforce(*args): pass
        async def mock_generate(*args): return "Mock Report"

        reporter.ceo_dashboard.enforce_strategic_vision = mock_enforce
        reporter.ceo_dashboard.generate_report = mock_generate
        """

code = code.replace("""        class DummyBot:
            def __init__(self):
                self.enabled = False
            async def start(self): pass
            def set_context(self, ctx): pass
            def set_performance_report(self, r): pass
            async def send_bet_signal(self, b): pass
            async def send_risk_alert(self, t, m): pass
            async def send_godmode_report(self, ctx): return True
            async def send_message(self, report): pass

        reporter = ReportingStage(bot_instance=DummyBot())""", replacement)

with open("tests/verify_pipeline.py", "w") as f:
    f.write(code)
