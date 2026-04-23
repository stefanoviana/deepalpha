#!/usr/bin/env python3
"""BUG 2: Fix cloud_trader.py _close_position - paper mode current_price + indentation."""

FILE = "/root/deepalpha-cloud/cloud_trader.py"

with open(FILE, "r") as f:
    content = f.read()

# The broken code block
OLD_BLOCK = """            if self.paper_mode:
                # PAPER: simulate close at current price
                logger.info("[user=%d] PAPER CLOSE: %s reason=%s", self.user_id, coin, reason)
                exit_price = current_price
            else:
                order = await self.exchange.create_order(
                self._symbol(coin), "market", close_side, pos["size"],
                params={"category": "linear", "reduceOnly": True},
            )
            exit_price = float(
                order.get("average") or order.get("price") or 0
            )
            filled = float(order.get("filled") or pos["size"])"""

NEW_BLOCK = """            if self.paper_mode:
                # PAPER: simulate close at current price
                ticker = await self.exchange.fetch_ticker(self._symbol(coin))
                current_price = float(ticker.get("last") or ticker.get("close") or pos["entry_price"])
                logger.info("[user=%d] PAPER CLOSE: %s reason=%s @ %.4f", self.user_id, coin, reason, current_price)
                exit_price = current_price
                filled = pos["size"]
            else:
                order = await self.exchange.create_order(
                    self._symbol(coin), "market", close_side, pos["size"],
                    params={"category": "linear", "reduceOnly": True},
                )
                exit_price = float(
                    order.get("average") or order.get("price") or 0
                )
                filled = float(order.get("filled") or pos["size"])"""

if OLD_BLOCK in content:
    content = content.replace(OLD_BLOCK, NEW_BLOCK)
    with open(FILE, "w") as f:
        f.write(content)
    print("BUG 2 FIXED: paper mode current_price + indentation in _close_position")
else:
    print("ERROR: Could not find the broken _close_position block")
    # Debug
    lines = content.split('\n')
    for idx, line in enumerate(lines):
        if 'PAPER CLOSE' in line or 'current_price' in line:
            print(f"  Line {idx+1}: {line.rstrip()}")
