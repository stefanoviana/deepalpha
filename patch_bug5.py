#!/usr/bin/env python3
"""BUG 5: 5-min cooldown after WIN, 30-min after LOSS in pro_trader.py."""

FILE = "/root/hyperlend-bot/pro_bot/pro_trader.py"

with open(FILE, "r") as f:
    content = f.read()

count = 0

# Fix 1: Main close path (live mode) - around line 1611
OLD_COOLDOWN_1 = """                daily_pnl += net_pnl
                # Cooldown after ANY close (prevents re-entry trap)
                coin_cooldowns[coin] = time.time() + 1800  # 30min cooldown
                if net_pnl < 0:
                    daily_losses += 1
                    last_loss_time = time.time()
                    log.info(f"[COOLDOWN] {coin} on cooldown for 30min after loss")
                else:
                    log.info(f"[COOLDOWN] {coin} on cooldown for 30min after profit (avoid re-entry)")"""

NEW_COOLDOWN_1 = """                daily_pnl += net_pnl
                # Cooldown: 5min after WIN, 30min after LOSS
                if net_pnl > 0:
                    coin_cooldowns[coin] = time.time() + 300   # 5min cooldown after win
                    log.info(f"[COOLDOWN] {coin} on cooldown for 5min after win")
                else:
                    coin_cooldowns[coin] = time.time() + 1800  # 30min cooldown after loss
                    daily_losses += 1
                    last_loss_time = time.time()
                    log.info(f"[COOLDOWN] {coin} on cooldown for 30min after loss")"""

if OLD_COOLDOWN_1 in content:
    content = content.replace(OLD_COOLDOWN_1, NEW_COOLDOWN_1, 1)
    count += 1
    print("  Fixed main close cooldown")
else:
    print("  WARNING: Could not find main close cooldown block")

# Fix 2: Paper mode close - line 1514
OLD_PAPER_CD = """                coin_cooldowns[coin] = time.time() + 1800
                if net_pnl < 0:
                    daily_losses += 1
                    last_loss_time = time.time()"""

NEW_PAPER_CD = """                if net_pnl > 0:
                    coin_cooldowns[coin] = time.time() + 300   # 5min after win
                else:
                    coin_cooldowns[coin] = time.time() + 1800  # 30min after loss
                if net_pnl < 0:
                    daily_losses += 1
                    last_loss_time = time.time()"""

if OLD_PAPER_CD in content:
    content = content.replace(OLD_PAPER_CD, NEW_PAPER_CD, 1)
    count += 1
    print("  Fixed paper mode cooldown")
else:
    print("  WARNING: Could not find paper mode cooldown block")

# Fix 3: Retry close path - line 1672
OLD_RETRY_CD = """                        coin_cooldowns[coin] = time.time() + 1800
                        if net_pnl < 0:
                            daily_losses += 1
                            last_loss_time = time.time()"""

NEW_RETRY_CD = """                        if net_pnl > 0:
                            coin_cooldowns[coin] = time.time() + 300   # 5min after win
                        else:
                            coin_cooldowns[coin] = time.time() + 1800  # 30min after loss
                        if net_pnl < 0:
                            daily_losses += 1
                            last_loss_time = time.time()"""

if OLD_RETRY_CD in content:
    content = content.replace(OLD_RETRY_CD, NEW_RETRY_CD, 1)
    count += 1
    print("  Fixed retry close cooldown")
else:
    print("  WARNING: Could not find retry close cooldown block")

with open(FILE, "w") as f:
    f.write(content)
print(f"BUG 5 FIXED: {count}/3 cooldown locations updated (5min win / 30min loss)")
