import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = Path("c:/Users/immor/OneDrive/Belgeler/Projelerim/bahis/data/bahis.duckdb")

def analyze():
    if not DB_PATH.exists():
        print(f"Error: {DB_PATH} not found.")
        return

    con = duckdb.connect(str(DB_PATH))
    
    # Query to join signals with match results
    query = """
    SELECT 
        s.match_id, 
        s.selection, 
        s.odds, 
        s.stake_pct,
        m.home_team,
        m.away_team,
        m.home_score, 
        m.away_score, 
        m.status
    FROM signals s
    JOIN matches m ON s.match_id = m.match_id
    WHERE m.status = 'finished'
    """
    
    df = con.execute(query).df()
    
    if df.empty:
        print("No finished matches with associated signals found.")
        return

    # Determine outcome
    def get_result(row):
        h, a = row['home_score'], row['away_score']
        if h > a: return 'home'
        if h < a: return 'away'
        return 'draw'

    df['actual_result'] = df.apply(get_result, axis=1)
    df['is_win'] = df['selection'] == df['actual_result']
    
    # Calculate PnL (assuming base bankroll of 10000 for calculation purposes)
    base_bankroll = 10000.0
    df['stake_amount'] = df['stake_pct'] * base_bankroll
    df['pnl'] = df.apply(lambda r: r['stake_amount'] * (r['odds'] - 1) if r['is_win'] else -r['stake_amount'], axis=1)
    
    print("\n--- Profitability Analysis Report ---")
    print(f"Total Matches Analyzed: {len(df)}")
    print(f"Wins: {df['is_win'].sum()}")
    print(f"Losses: {len(df) - df['is_win'].sum()}")
    print(f"Win Rate: {df['is_win'].mean():.2%}")
    print(f"Total PnL: {df['pnl'].sum():.2f}")
    print(f"ROI: {df['pnl'].sum() / df['stake_amount'].sum():.2%}")
    
    print("\nTop 5 Most Profitable Trades:")
    print(df.sort_values('pnl', ascending=False).head(5)[['match_id', 'home_team', 'away_team', 'selection', 'odds', 'pnl']])

    con.close()

if __name__ == "__main__":
    analyze()
