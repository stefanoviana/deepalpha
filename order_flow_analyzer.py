"""
Order Flow Imbalance (OFI) Analyzer — Cross-Venue Module
=========================================================
Analyzes L2 orderbook data from Hyperliquid and Binance Futures to produce
directional signals based on order flow imbalance, VPIN, and cross-venue
divergence detection.

Standalone module — no dependencies on pro_trader.py.
Requires: requests
"""

import time
import math
import logging
import requests
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger("order_flow_analyzer")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CACHE_TTL_SECONDS = 5
OFI_LEVELS = 10  # top N levels of the book to consider
VPIN_WINDOW = 50  # number of buckets for VPIN calculation
AGGRESSIVENESS_WINDOW = 20  # snapshots to track for aggressiveness
REQUEST_TIMEOUT = 4  # seconds

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
BINANCE_DEPTH_URL = "https://fapi.binance.com/fapi/v1/depth"

# Thresholds
OFI_STRONG_THRESHOLD = 0.25  # |OFI| above this = meaningful imbalance
DIVERGENCE_BONUS_THRESHOLD = 0.15  # min gap between venues to trigger bonus
VPIN_HIGH_THRESHOLD = 0.6  # VPIN above this = informed trading likely
CONFIDENCE_OFI_WEIGHT = 0.35
CONFIDENCE_VPIN_WEIGHT = 0.25
CONFIDENCE_CROSS_WEIGHT = 0.25
CONFIDENCE_AGGR_WEIGHT = 0.15


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BookSnapshot:
    """Single L2 orderbook snapshot (top N levels)."""
    bids: List[Tuple[float, float]] = field(default_factory=list)  # (price, size)
    asks: List[Tuple[float, float]] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class OFIResult:
    """Full order-flow signal output."""
    ofi_hl: float = 0.0
    ofi_binance: float = 0.0
    cross_venue_score: float = 0.0
    vpin: float = 0.0
    direction: str = "NEUTRAL"
    confidence: float = 0.0
    aggressiveness: float = 0.0


# ---------------------------------------------------------------------------
# Module-level state (per-coin history for VPIN / aggressiveness)
# ---------------------------------------------------------------------------
_cache: Dict[str, Tuple[float, OFIResult]] = {}
_ofi_history: Dict[str, deque] = {}          # coin -> deque of (ts, ofi)
_book_history: Dict[str, deque] = {}         # coin -> deque of BookSnapshot
_volume_buckets: Dict[str, deque] = {}       # coin -> deque of (buy_vol, sell_vol)


def _get_history(store: dict, coin: str, maxlen: int = 100) -> deque:
    if coin not in store:
        store[coin] = deque(maxlen=maxlen)
    return store[coin]


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def _fetch_hl_book(coin: str) -> Optional[BookSnapshot]:
    """Fetch L2 orderbook from Hyperliquid REST API."""
    try:
        resp = requests.post(
            HL_INFO_URL,
            json={"type": "l2Book", "coin": coin},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        levels = data.get("levels", [[], []])
        bids_raw = levels[0] if len(levels) > 0 else []
        asks_raw = levels[1] if len(levels) > 1 else []

        bids = [(float(b["px"]), float(b["sz"])) for b in bids_raw[:OFI_LEVELS]]
        asks = [(float(a["px"]), float(a["sz"])) for a in asks_raw[:OFI_LEVELS]]

        return BookSnapshot(bids=bids, asks=asks, timestamp=time.time())
    except Exception as e:
        logger.warning("HL book fetch failed for %s: %s", coin, e)
        return None


def _fetch_binance_book(coin: str) -> Optional[BookSnapshot]:
    """Fetch L2 orderbook from Binance Futures REST API."""
    try:
        symbol = coin.upper() + "USDT"
        resp = requests.get(
            BINANCE_DEPTH_URL,
            params={"symbol": symbol, "limit": OFI_LEVELS},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        bids = [(float(b[0]), float(b[1])) for b in data.get("bids", [])[:OFI_LEVELS]]
        asks = [(float(a[0]), float(a[1])) for a in data.get("asks", [])[:OFI_LEVELS]]

        return BookSnapshot(bids=bids, asks=asks, timestamp=time.time())
    except Exception as e:
        logger.warning("Binance book fetch failed for %s: %s", coin, e)
        return None


# ---------------------------------------------------------------------------
# OFI calculation
# ---------------------------------------------------------------------------
def _calc_ofi(book: BookSnapshot) -> float:
    """
    Order Flow Imbalance from a single book snapshot.
    OFI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    Range: -1 (all sells) to +1 (all buys).
    """
    bid_vol = sum(sz for _, sz in book.bids)
    ask_vol = sum(sz for _, sz in book.asks)
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


# ---------------------------------------------------------------------------
# VPIN calculation
# ---------------------------------------------------------------------------
def _classify_trade_direction(prev_book: Optional[BookSnapshot],
                              curr_book: BookSnapshot) -> Tuple[float, float]:
    """
    Estimate buy vs sell volume using the tick rule on mid-price movement
    combined with volume delta between snapshots.

    Returns (estimated_buy_volume, estimated_sell_volume).
    """
    if prev_book is None or not prev_book.bids or not prev_book.asks:
        # No previous data — split evenly
        bid_vol = sum(sz for _, sz in curr_book.bids)
        ask_vol = sum(sz for _, sz in curr_book.asks)
        total = bid_vol + ask_vol
        return (total / 2, total / 2) if total > 0 else (0.0, 0.0)

    prev_mid = (prev_book.bids[0][0] + prev_book.asks[0][0]) / 2 if prev_book.bids and prev_book.asks else 0
    curr_mid = (curr_book.bids[0][0] + curr_book.asks[0][0]) / 2 if curr_book.bids and curr_book.asks else 0

    bid_vol = sum(sz for _, sz in curr_book.bids)
    ask_vol = sum(sz for _, sz in curr_book.asks)
    total = bid_vol + ask_vol
    if total == 0:
        return (0.0, 0.0)

    # Tick rule: if mid moved up, net flow was buying; if down, selling
    if curr_mid > prev_mid:
        buy_frac = 0.65
    elif curr_mid < prev_mid:
        buy_frac = 0.35
    else:
        buy_frac = 0.50

    # Adjust with book imbalance
    ofi = _calc_ofi(curr_book)
    buy_frac = max(0.1, min(0.9, buy_frac + ofi * 0.15))

    buy_vol = total * buy_frac
    sell_vol = total * (1 - buy_frac)
    return (buy_vol, sell_vol)


def _calc_vpin(coin: str, curr_book: BookSnapshot) -> float:
    """
    Volume-synchronized Probability of Informed Trading.
    VPIN = mean(|buy_vol - sell_vol| / total_vol) over rolling window.
    Range: 0 (balanced) to 1 (fully one-sided / informed).
    """
    history = _get_history(_book_history, coin, maxlen=VPIN_WINDOW + 1)
    prev_book = history[-1] if history else None
    history.append(curr_book)

    buy_vol, sell_vol = _classify_trade_direction(prev_book, curr_book)
    total = buy_vol + sell_vol

    buckets = _get_history(_volume_buckets, coin, maxlen=VPIN_WINDOW)
    buckets.append((buy_vol, sell_vol))

    if len(buckets) < 3:
        # Not enough data yet
        return 0.0 if total == 0 else abs(buy_vol - sell_vol) / total

    vpin_sum = 0.0
    vpin_total = 0.0
    for bv, sv in buckets:
        t = bv + sv
        if t > 0:
            vpin_sum += abs(bv - sv) / t
            vpin_total += 1

    return vpin_sum / vpin_total if vpin_total > 0 else 0.0


# ---------------------------------------------------------------------------
# Cross-venue divergence
# ---------------------------------------------------------------------------
def _calc_cross_venue_score(ofi_hl: float, ofi_binance: float) -> float:
    """
    Combined cross-venue score.
    cross_venue_score = hl_ofi * 0.4 + binance_ofi * 0.4 + divergence_bonus * 0.2

    Divergence bonus is positive when both venues agree (reinforcing),
    and negative when they disagree (conflicting).
    """
    divergence = abs(ofi_hl - ofi_binance)

    if divergence < DIVERGENCE_BONUS_THRESHOLD:
        # Venues agree — bonus in the direction of consensus
        avg_dir = (ofi_hl + ofi_binance) / 2
        bonus = math.copysign(min(divergence + 0.3, 1.0), avg_dir)
    else:
        # Venues disagree — reduce confidence, bonus toward zero
        bonus = 0.0

    score = ofi_hl * 0.4 + ofi_binance * 0.4 + bonus * 0.2
    return max(-1.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Aggressiveness detection
# ---------------------------------------------------------------------------
def _calc_aggressiveness(coin: str, curr_book: BookSnapshot) -> float:
    """
    Detect aggressive (taker) flow by tracking bid/ask top-of-book
    volume depletion between snapshots.

    A sudden drop in best-bid or best-ask size implies a market order
    consumed liquidity (aggressive taker).

    Returns aggressiveness score 0.0 to 1.0.
    """
    history = _get_history(_ofi_history, coin, maxlen=AGGRESSIVENESS_WINDOW)

    if not history:
        history.append((time.time(), curr_book, 0.0))
        return 0.0

    _, prev_book, _ = history[-1]

    aggr_score = 0.0

    # Check bid side depletion (aggressive selling hitting bids)
    if prev_book.bids and curr_book.bids:
        prev_best_bid_sz = prev_book.bids[0][1]
        curr_best_bid_sz = curr_book.bids[0][1]
        if prev_best_bid_sz > 0:
            bid_depletion = max(0, prev_best_bid_sz - curr_best_bid_sz) / prev_best_bid_sz
        else:
            bid_depletion = 0.0
    else:
        bid_depletion = 0.0

    # Check ask side depletion (aggressive buying lifting asks)
    if prev_book.asks and curr_book.asks:
        prev_best_ask_sz = prev_book.asks[0][1]
        curr_best_ask_sz = curr_book.asks[0][1]
        if prev_best_ask_sz > 0:
            ask_depletion = max(0, prev_best_ask_sz - curr_best_ask_sz) / prev_best_ask_sz
        else:
            ask_depletion = 0.0
    else:
        ask_depletion = 0.0

    aggr_score = max(bid_depletion, ask_depletion)

    # Smooth with recent history: use exponential moving average
    recent_scores = [s for _, _, s in history]
    if recent_scores:
        alpha = 0.3
        ema = recent_scores[-1]
        ema = alpha * aggr_score + (1 - alpha) * ema
        aggr_score = ema

    history.append((time.time(), curr_book, aggr_score))
    return max(0.0, min(1.0, aggr_score))


# ---------------------------------------------------------------------------
# Direction & confidence
# ---------------------------------------------------------------------------
def _determine_direction(cross_venue_score: float, vpin: float,
                         aggressiveness: float) -> Tuple[str, float]:
    """
    Determine trade direction and confidence from all sub-signals.

    Returns (direction, confidence).
    """
    abs_score = abs(cross_venue_score)

    # Confidence is a weighted blend of signal strengths
    confidence = (
        CONFIDENCE_OFI_WEIGHT * abs_score
        + CONFIDENCE_VPIN_WEIGHT * vpin
        + CONFIDENCE_CROSS_WEIGHT * abs_score  # cross reinforces OFI
        + CONFIDENCE_AGGR_WEIGHT * aggressiveness
    )
    confidence = max(0.0, min(1.0, confidence))

    # Direction
    if abs_score < 0.05 or confidence < 0.15:
        direction = "NEUTRAL"
    elif cross_venue_score > 0:
        direction = "LONG"
    else:
        direction = "SHORT"

    return direction, round(confidence, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_order_flow_signal(coin: str) -> Dict:
    """
    Main entry point. Fetches orderbooks from Hyperliquid and Binance,
    computes OFI, VPIN, cross-venue score, aggressiveness, and returns
    a unified signal dict.

    Results are cached for CACHE_TTL_SECONDS to avoid API spam.

    Parameters
    ----------
    coin : str
        Asset symbol, e.g. "BTC", "ETH", "SOL".

    Returns
    -------
    dict with keys:
        ofi_hl          : float (-1 to 1)   Hyperliquid OFI
        ofi_binance     : float (-1 to 1)   Binance OFI
        cross_venue_score: float (-1 to 1)  Combined score
        vpin            : float (0 to 1)    Informed trading probability
        direction       : str               'LONG' / 'SHORT' / 'NEUTRAL'
        confidence      : float (0 to 1)    Signal confidence
        aggressiveness  : float (0 to 1)    Taker aggressiveness score
    """
    coin_upper = coin.upper()
    now = time.time()

    # --- Cache check ---
    if coin_upper in _cache:
        cached_ts, cached_result = _cache[coin_upper]
        if now - cached_ts < CACHE_TTL_SECONDS:
            return _result_to_dict(cached_result)

    # --- Fetch orderbooks ---
    hl_book = _fetch_hl_book(coin_upper)
    binance_book = _fetch_binance_book(coin_upper)

    if hl_book is None and binance_book is None:
        logger.error("Both venues failed for %s — returning neutral", coin_upper)
        result = OFIResult()
        _cache[coin_upper] = (now, result)
        return _result_to_dict(result)

    # --- Compute OFI per venue ---
    ofi_hl = _calc_ofi(hl_book) if hl_book else 0.0
    ofi_binance = _calc_ofi(binance_book) if binance_book else 0.0

    # --- Cross-venue score ---
    if hl_book and binance_book:
        cross_score = _calc_cross_venue_score(ofi_hl, ofi_binance)
    elif hl_book:
        cross_score = ofi_hl  # single venue fallback
    else:
        cross_score = ofi_binance

    # --- VPIN (use HL book as primary, fallback to Binance) ---
    primary_book = hl_book or binance_book
    vpin = _calc_vpin(coin_upper, primary_book)

    # --- Aggressiveness ---
    aggressiveness = _calc_aggressiveness(coin_upper, primary_book)

    # --- Direction & confidence ---
    direction, confidence = _determine_direction(cross_score, vpin, aggressiveness)

    result = OFIResult(
        ofi_hl=round(ofi_hl, 4),
        ofi_binance=round(ofi_binance, 4),
        cross_venue_score=round(cross_score, 4),
        vpin=round(vpin, 4),
        direction=direction,
        confidence=confidence,
        aggressiveness=round(aggressiveness, 4),
    )

    _cache[coin_upper] = (now, result)
    return _result_to_dict(result)


def _result_to_dict(r: OFIResult) -> Dict:
    return {
        "ofi_hl": r.ofi_hl,
        "ofi_binance": r.ofi_binance,
        "cross_venue_score": r.cross_venue_score,
        "vpin": r.vpin,
        "direction": r.direction,
        "confidence": r.confidence,
        "aggressiveness": r.aggressiveness,
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    coins = ["BTC", "ETH", "SOL"]
    for c in coins:
        sig = get_order_flow_signal(c)
        print(f"\n{'='*50}")
        print(f"  {c} Order Flow Signal")
        print(f"{'='*50}")
        for k, v in sig.items():
            print(f"  {k:>22s}: {v}")
