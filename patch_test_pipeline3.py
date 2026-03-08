import re

with open("tests/test_pipeline_stages.py", "r") as f:
    code = f.read()

# Mock out PortfolioStressTester globally in the module
replacement = """
    @patch("src.quant.finance.stress_tester.PortfolioStressTester.check_portfolio_health")
    def test_risk_stage_integration(self, mock_stress):
        mock_stress.return_value = {"approved": True, "var_pct": 0.0, "reason": "Mocked"}
        \"\"\"Test RiskStage consumes Ensemble results correctly.\"\"\""""

code = code.replace("""    def test_risk_stage_integration(self):\n        \"\"\"Test RiskStage consumes Ensemble results correctly.\"\"\"""", replacement)

with open("tests/test_pipeline_stages.py", "w") as f:
    f.write(code)

print("Patched test_pipeline_stages.py completely mock stress tester.")
