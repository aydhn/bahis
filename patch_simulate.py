import re

with open("src/system/digital_twin.py", "r") as f:
    content = f.read()

new_method = """    async def _simulate_single_match(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"
        Simulates a single match through the decision engine.
        Returns the virtual outcome (Win/Loss/PnL).
        \"\"\"
        if not self.pipeline:
            return {}

        res = await self.pipeline.run_once({"match_data": match_data})

        return {
            "match_id": match_data.get("match_id"),
            "decision": res.get("decision", "SKIP"),
            "result": res.get("result"),
            "pnl": res.get("pnl", 0.0),
            "won": res.get("won", False)
        }"""

pattern = r'    async def _simulate_single_match\(self, match_data: Dict\[str, Any\]\) -> Dict\[str, Any\]:.*'
replaced = re.sub(pattern, new_method, content, flags=re.DOTALL)

with open("src/system/digital_twin.py", "w") as f:
    f.write(replaced)
