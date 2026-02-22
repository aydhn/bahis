
print("Checking lines...")
try:
    with open('c:/Users/immor/OneDrive/Belgeler/Projelerim/bahis/bahis.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"Total lines: {len(lines)}")
        if len(lines) > 1318:
            print(f"Line 1318 (content): {lines[1317].rstrip()}")
            print(f"Line 1319 (content): {lines[1318].rstrip()}")
        if len(lines) > 6709:
            print(f"Line 6708 (content): {lines[6707].rstrip()}")
            print(f"Line 6709 (content): {lines[6708].rstrip()}")
            print(f"Line 6710 (content): {lines[6709].rstrip()}")
except Exception as e:
    print(f"Error: {e}")
