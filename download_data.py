"""
DeepAlpha — Data Downloader
Downloads 1h candle data from Hyperliquid API for the top coins.
No API key required — Hyperliquid's info API is public.

Usage:
    python download_data.py
"""

import json
import os
import time
import requests
import config

# Hyperliquid public info endpoint
HL_INFO_URL = "https://api.hyperliquid.xyz/info"

# How many days of history to download
DAYS_BACK = 365


def get_candles(coin: str, interval: str = "1h", days: int = DAYS_BACK) -> list[dict]:
    """
    Fetch historical candle data from Hyperliquid.

    Parameters
    ----------
    coin : str — e.g. "BTC"
    interval : str — candle interval (e.g. "1h")
    days : int — how many days of history

    Returns
    -------
    List of candle dicts with keys: t, o, h, l, c, v
    """
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 3600 * 1000)

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }

    resp = requests.post(HL_INFO_URL, json=payload, timeout=30)
    resp.raise_for_status()
    raw = resp.json()

    # Normalise to a clean format
    candles = []
    for c in raw:
        candles.append({
            "t": c["t"],               # timestamp ms
            "o": float(c["o"]),         # open
            "h": float(c["h"]),         # high
            "l": float(c["l"]),         # low
            "c": float(c["c"]),         # close
            "v": float(c["v"]),         # volume
        })

    return candles


def download_all() -> None:
    """Download candle data for all configured coins and save to data/."""
    os.makedirs(config.DATA_DIR, exist_ok=True)

    print(f"Downloading {DAYS_BACK} days of 1h candles for {len(config.COINS)} coins...")
    print(f"Saving to: {config.DATA_DIR}/\n")

    for i, coin in enumerate(config.COINS, 1):
        try:
            candles = get_candles(coin, config.CANDLE_INTERVAL, DAYS_BACK)
            path = os.path.join(config.DATA_DIR, f"{coin}_1h.json")
            with open(path, "w") as f:
                json.dump(candles, f)
            print(f"  [{i:2d}/{len(config.COINS)}] {coin:6s} — {len(candles):,} candles saved")
        except Exception as e:
            print(f"  [{i:2d}/{len(config.COINS)}] {coin:6s} — ERROR: {e}")

        # Small delay to be polite to the API
        time.sleep(0.5)

    print("\nDone! Data saved to data/ directory.")


if __name__ == "__main__":
    download_all()
