import sqlite3
import json
from datetime import datetime

DB_PATH = 'c:/Users/immor/OneDrive/Belgeler/Projelerim/bahis/data/events.db'

def analyze_events():
    conn = sqlite3.connect(DB_PATH)
    
    # bet_placed ve bet_settled olaylarını al
    rows = conn.execute("""
        SELECT event_type, data, timestamp 
        FROM events 
        WHERE event_type IN ('bet_placed', 'bet_settled') 
        ORDER BY timestamp ASC
    """).fetchall()
    
    placed = {}
    settled = []
    
    for row_type, data_json, ts in rows:
        data = json.loads(data_json)
        match_id = data.get('match_id')
        if not match_id: continue
        
        if row_type == 'bet_placed':
            placed[match_id] = {'data': data, 'ts': ts}
        elif row_type == 'bet_settled':
            settled.append({'match_id': match_id, 'data': data, 'ts': ts})
            
    print(f"Total placed: {len(placed)}")
    print(f"Total settled: {len(settled)}")
    
    # Tutarsızlıkları bul: Placed ama Settled değil
    unsettled = []
    for mid, p_info in placed.items():
        if not any(s['match_id'] == mid for s in settled):
            unsettled.append(mid)
            
    print(f"Unsettled (Placed but not settled): {len(unsettled)}")
    if unsettled:
        print("Sample unsettled match_ids:", unsettled[:10])
        
    # Kârlılık analizi
    total_stake = 0
    total_payout = 0
    for s in settled:
        stake = s['data'].get('stake', 0) or s['data'].get('stake_amount', 0)
        payout = s['data'].get('payout', 0)
        total_stake += stake
        total_payout += payout
        
    print(f"Total Stake: {total_stake}")
    print(f"Total Payout: {total_payout}")
    if total_stake > 0:
        print(f"ROI: {(total_payout - total_stake) / total_stake:.2%}")
        print(f"Profit: {total_payout - total_stake}")

if __name__ == "__main__":
    analyze_events()
