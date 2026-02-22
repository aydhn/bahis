from pathlib import Path
import os

def read_last_lines(file_path, n=50):
    try:
        p = Path(file_path)
        if not p.exists():
            return [f"File not found: {file_path}\n"]
        
        # Simple implementation for text files
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            return lines[-n:]
    except Exception as e:
        return [f"Error reading {file_path}: {e}\n"]

print("=== bot_2026-02-19.log ===")
print("".join(read_last_lines("logs/bot_2026-02-19.log", 50)))
print("\n=== error.log ===")
print("".join(read_last_lines("logs/error.log", 50)))
