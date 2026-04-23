#!/usr/bin/env python3
"""BUG 4: Fix backtest overwriting global MAX_POSITIONS and LEVERAGE in app.py."""

FILE = "/root/deepalpha-cloud/app.py"

with open(FILE, "r") as f:
    content = f.read()

count = 0

# Replace the global assignments with local variables
OLD1 = "    MAX_POSITIONS = body.max_positions"
NEW1 = "    bt_max_positions = body.max_positions"
if OLD1 in content:
    content = content.replace(OLD1, NEW1, 1)
    count += 1
else:
    print("ERROR: Could not find MAX_POSITIONS = body.max_positions")

OLD2 = "    LEVERAGE = body.leverage"
NEW2 = "    bt_leverage = body.leverage"
if OLD2 in content:
    content = content.replace(OLD2, NEW2, 1)
    count += 1
else:
    print("ERROR: Could not find LEVERAGE = body.leverage")

# Replace usages within the backtest function
# POSITION_SIZE uses LEVERAGE
OLD3 = "    POSITION_SIZE = margin_per_trade * LEVERAGE"
NEW3 = "    POSITION_SIZE = margin_per_trade * bt_leverage"
if OLD3 in content:
    content = content.replace(OLD3, NEW3, 1)
    count += 1

# MAX_POSITIONS used in position checks
OLD4 = "        if len(open_positions) < MAX_POSITIONS:"
NEW4 = "        if len(open_positions) < bt_max_positions:"
if OLD4 in content:
    content = content.replace(OLD4, NEW4, 1)
    count += 1

OLD5 = "            slots = MAX_POSITIONS - len(open_positions)"
NEW5 = "            slots = bt_max_positions - len(open_positions)"
if OLD5 in content:
    content = content.replace(OLD5, NEW5, 1)
    count += 1

# Result dict references
OLD6 = '        "max_positions": MAX_POSITIONS,'
NEW6 = '        "max_positions": bt_max_positions,'
if OLD6 in content:
    content = content.replace(OLD6, NEW6, 1)
    count += 1

OLD7 = '        "leverage": LEVERAGE,'
NEW7 = '        "leverage": bt_leverage,'
if OLD7 in content:
    content = content.replace(OLD7, NEW7, 1)
    count += 1

with open(FILE, "w") as f:
    f.write(content)
print(f"BUG 4 FIXED: {count}/7 replacements - backtest uses local bt_max_positions/bt_leverage")
