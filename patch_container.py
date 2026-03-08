import re

with open("src/system/container.py", "r") as f:
    code = f.read()

# Fix syntax error properly
code = re.sub(r'            elif name == "active_agent":.*?elif name == "smart_money":.*?self\._services\["smart_money"\] = SmartMoneyDetector\(\)',
"""            elif name == "active_agent":
                from src.core.active_inference_agent import ActiveInferenceAgent
                self._services["active_agent"] = ActiveInferenceAgent()

            elif name == "smart_money":
                from src.extensions.smart_money import SmartMoneyDetector
                self._services["smart_money"] = SmartMoneyDetector()""", code, flags=re.DOTALL)

with open("src/system/container.py", "w") as f:
    f.write(code)

print("Patched container.py")
