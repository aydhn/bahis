import polars as pl
from src.quant.physics.path_signature_engine import PathSignatureEngine

pse = PathSignatureEngine()
df = pl.DataFrame({
    "match_id": ["m1"],
    "home_odds": [2.0],
    "draw_odds": [3.0],
    "away_odds": [4.0]
})
res = pse.extract(df)
print(res)
