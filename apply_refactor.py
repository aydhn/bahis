
new_loop_body = """
        try:
            # Orchestrator'ı çalıştır
            logger.info("🔄 [Cycle] Yeni döngü başlıyor...")
            portfolio = await orchestrator.run_cycle()
            
            # Adaptive Sleep Logic
            if portfolio and len(portfolio) > 0:
                logger.info(f"✅ {len(portfolio)} bahis alındı. Hızlı döngü (15s).")
                await _await_with_timeout(asyncio.sleep(15), 20)
            else:
                logger.info("💤 Bahis yok. Bekleme moduna geçiliyor (60s).")
                await _await_with_timeout(asyncio.sleep(60), 70)
                
        except Exception as e:
            logger.error(f"❌ [Cycle] Kritik Hata: {e}")
            await asyncio.sleep(5) # Error recovery backlog

    logger.info("🛑 [System] Analiz döngüsü sonlandırıldı.")

"""

try:
    with open('c:/Users/immor/OneDrive/Belgeler/Projelerim/bahis/bahis.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Check bounds again just in case
    if "while not shutdown.is_set():" not in lines[1317]:
         print("Error: Line 1318 mismatch")
         exit(1)
    if "@app.command()" not in lines[6708]:
         print("Error: Line 6709 mismatch")
         exit(1)

    # Slice and dice
    final_lines = lines[:1318] + [new_loop_body] + lines[6708:]
    
    with open('c:/Users/immor/OneDrive/Belgeler/Projelerim/bahis/bahis.py', 'w', encoding='utf-8') as f:
        f.writelines(final_lines)
    
    print("Successfully refactored bahis.py")

except Exception as e:
    print(f"Error: {e}")
