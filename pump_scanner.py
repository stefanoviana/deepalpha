"""
DeepAlpha — Pump Detection & Riding System
============================================
Scans ALL Bybit USDT perpetual pairs for sudden volume/price spikes,
rides the pump long, and shorts the dump after exhaustion.

Also monitors Bybit announcements API for new listings.

Designed to run as a separate thread/process alongside the main AI bot,
with its own risk budget (PUMP_RISK_BUDGET_PCT of total equity).

Usage:
    from pump_scanner import PumpScanner

    scanner = PumpScanner(exchange_client)  # ccxt bybit instance
    scanner.start()  # launches background scanning thread
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger("pump_scanner")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Pump Scanner Settings
# ═══════════════════════════════════════════════════════════════════════════

# --- Detection thresholds ---
SCAN_INTERVAL_SEC = int(os.getenv("PUMP_SCAN_INTERVAL", "3"))         # scan every 3s
VOLUME_SPIKE_MULT = float(os.getenv("PUMP_VOL_SPIKE_MULT", "5.0"))    # 5x normal volume
PRICE_SPIKE_PCT = float(os.getenv("PUMP_PRICE_SPIKE_PCT", "0.03"))    # 3% move in window
PRICE_WINDOW_CANDLES = int(os.getenv("PUMP_PRICE_WINDOW", "5"))       # 5x 1m candles = 5min
EWMA_SPAN = int(os.getenv("PUMP_EWMA_SPAN", "20"))                   # 20-period EWMA for baseline volume
MIN_DOLLAR_VOLUME = float(os.getenv("PUMP_MIN_DOLLAR_VOL", "500000")) # ignore illiquid coins

# --- Fakeout filter ---
CONFIRM_CANDLES = int(os.getenv("PUMP_CONFIRM_CANDLES", "3"))         # need 3 consecutive up candles
MIN_RSI_ENTRY = float(os.getenv("PUMP_MIN_RSI_ENTRY", "60"))         # RSI must be > 60 (momentum)
MAX_RSI_ENTRY = float(os.getenv("PUMP_MAX_RSI_ENTRY", "85"))         # RSI < 85 (not already exhausted)
MIN_BUY_RATIO = float(os.getenv("PUMP_MIN_BUY_RATIO", "0.65"))      # 65%+ buy-side taker volume

# --- Position management ---
PUMP_LEVERAGE = int(os.getenv("PUMP_LEVERAGE", "5"))
PUMP_RISK_BUDGET_PCT = float(os.getenv("PUMP_RISK_BUDGET_PCT", "0.05"))  # 5% of equity for pump trades
PUMP_MAX_POSITIONS = int(os.getenv("PUMP_MAX_POSITIONS", "2"))
PUMP_SL_ATR_MULT = float(os.getenv("PUMP_SL_ATR_MULT", "1.5"))       # SL = 1.5x ATR below entry
PUMP_TP1_PCT = float(os.getenv("PUMP_TP1_PCT", "0.05"))              # +5%  take 40%
PUMP_TP2_PCT = float(os.getenv("PUMP_TP2_PCT", "0.10"))              # +10% take 30%
PUMP_TP3_PCT = float(os.getenv("PUMP_TP3_PCT", "0.20"))              # +20% take remaining 30%
PUMP_TRAILING_PCT = float(os.getenv("PUMP_TRAILING_PCT", "0.03"))     # 3% trailing after TP2

# --- Dump short settings ---
SHORT_RSI_THRESHOLD = float(os.getenv("PUMP_SHORT_RSI", "80"))        # RSI > 80 = overbought
SHORT_VOL_DECLINE_PCT = float(os.getenv("PUMP_SHORT_VOL_DECLINE", "0.40"))  # volume drops 40%
SHORT_FUNDING_EXTREME = float(os.getenv("PUMP_SHORT_FUNDING", "0.001"))     # funding > 0.1% = extreme
SHORT_SL_ATR_MULT = float(os.getenv("PUMP_SHORT_SL_ATR", "2.0"))
SHORT_TP_PCT = float(os.getenv("PUMP_SHORT_TP_PCT", "0.05"))          # 5% TP on short

# --- New listing detection ---
LISTING_CHECK_INTERVAL = int(os.getenv("PUMP_LISTING_CHECK", "30"))   # check every 30s
LISTING_BUY_DELAY_SEC = int(os.getenv("PUMP_LISTING_DELAY", "5"))     # wait 5s after detection
LISTING_RISK_PCT = float(os.getenv("PUMP_LISTING_RISK", "0.02"))      # 2% equity per listing trade

# --- Telegram alerts ---
PUMP_TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
PUMP_TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- Cooldown per coin (avoid re-entering same pump) ---
PUMP_COOLDOWN_SEC = int(os.getenv("PUMP_COOLDOWN_SEC", "1800"))       # 30 min cooldown

# --- Daily loss circuit breaker ---
MAX_DAILY_PUMP_LOSS = float(os.getenv("PUMP_MAX_DAILY_LOSS", "-50"))  # stop after $50 daily loss

# --- Minimum notional (Bybit minimum) ---
MIN_NOTIONAL = float(os.getenv("PUMP_MIN_NOTIONAL", "5.0"))

# --- Market cache TTL ---
MARKET_CACHE_TTL_SEC = int(os.getenv("PUMP_MARKET_CACHE_TTL", "300"))  # 5 minutes


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PumpSignal:
    """Detected pump event."""
    coin: str
    signal_type: str              # "pump_long", "dump_short", "new_listing"
    detected_at: float            # timestamp
    price_at_detection: float
    volume_ratio: float           # current vol / baseline vol
    rsi: float
    atr: float
    confidence: float             # 0-1 composite score
    metadata: dict = field(default_factory=dict)


@dataclass
class PumpPosition:
    """Active pump trade being managed."""
    coin: str
    side: str                     # "long" or "short"
    entry_price: float
    quantity: float
    original_quantity: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    trailing_active: bool = False
    trailing_high: float = 0.0
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    opened_at: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# CORE SCANNER CLASS
# ═══════════════════════════════════════════════════════════════════════════

class PumpScanner:
    """
    Real-time pump detection and execution engine.

    Runs as a background thread that:
    1. Fetches 1m tickers for all USDT perps every SCAN_INTERVAL_SEC
    2. Computes volume/price anomaly scores
    3. Filters fakeouts via RSI, buy-ratio, consecutive candles
    4. Opens long on confirmed pump, short on exhaustion
    5. Manages positions with partial TP + trailing stop
    """

    def __init__(self, ccxt_client, telegram_fn=None):
        """
        Parameters
        ----------
        ccxt_client : ccxt.bybit instance (already authenticated + load_markets)
        telegram_fn : optional callable(message: str) for alerts
        """
        self.client = ccxt_client
        self.telegram_fn = telegram_fn
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Thread lock for shared state (Fix #5)
        self._lock = threading.Lock()

        # Blacklist: stock/pre-market tokens that aren't real crypto perps
        self._blacklist = {
            'TSLA', 'TSM', 'INTC', 'HOOD', 'CHIP', 'OPG', 'AAPL', 'AMZN', 'GOOG', 'GOOGL',
            'MSFT', 'NVDA', 'META', 'NFLX', 'AMD', 'COIN', 'MSTR', 'PLTR', 'UBER',
            'SQ', 'PYPL', 'SHOP', 'SNOW', 'CRWD', 'NET', 'DDOG', 'ZS',
            'BABA', 'DIS', 'BA', 'JPM', 'V', 'MA', 'WMT', 'PFE', 'KO', 'PEP',
            'COST', 'CSCO', 'ORCL', 'CRM', 'ABNB', 'SNAP', 'PINS', 'ROKU', 'SQ',
            'TSLAX', 'GOOGLX', 'AAPLX', 'AMZNX', 'MSFTX', 'NVDAX', 'METAX',
        }

        # Persistent set of already-seen listings (survives restarts)
        self._seen_listings_file = os.path.join(os.path.dirname(__file__), "pump_seen_listings.json")
        self._seen_listings: set[str] = set()
        try:
            import json as _json
            with open(self._seen_listings_file, "r") as f:
                self._seen_listings = set(_json.load(f))
        except Exception:
            pass

        # State
        self.volume_baselines: dict[str, list[float]] = defaultdict(list)  # coin -> rolling volumes
        self.price_history: dict[str, list[float]] = defaultdict(list)     # coin -> recent closes
        self.pump_positions: dict[str, PumpPosition] = {}                  # coin -> active position
        self.cooldowns: dict[str, float] = {}                              # coin -> cooldown_until timestamp
        self.known_listings: set[str] = set()                              # already-seen listing symbols
        self._daily_pump_pnl: float = 0.0
        self._daily_pump_pnl_date: str = ""

        # Non-blocking new listing handling (Fix #4)
        self._pending_listing: Optional[dict] = None  # {"coin": str, "time": float}

        # Market cache (Fix #7)
        self._markets_last_loaded: float = 0.0

        # All USDT perp symbols (populated on start)
        self._all_symbols: list[str] = []

    # ─── Lifecycle ──────────────────────────────────────────────────────

    def _safe_fetch_ohlcv(self, symbol, *args, **kwargs):
        try:
            return self.client.fetch_ohlcv(symbol, *args, **kwargs)
        except Exception:
            return []

    def start(self):
        """Start the pump scanner in a background thread."""
        if self._running:
            logger.warning("PumpScanner already running")
            return
        self._running = True
        self._load_all_symbols()
        self._thread = threading.Thread(target=self._main_loop, daemon=True, name="PumpScanner")
        self._thread.start()
        logger.info(f"PumpScanner started — monitoring {len(self._all_symbols)} USDT perp pairs")
        self._alert(f"PUMP SCANNER STARTED - monitoring {len(self._all_symbols)} pairs")

    def stop(self):
        """Stop the scanner."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("PumpScanner stopped")

    # ─── Symbol loading ────────────────────────────────────────────────

    def _load_all_symbols(self):
        """Load all active USDT linear perpetual symbols from Bybit."""
        markets = self.client.markets or self.client.load_markets()
        self._all_symbols = []
        for sym, info in (markets or {}).items():
            if (info.get("linear") and
                info.get("active") and
                info.get("quote") == "USDT" and
                info.get("type") == "swap"):
                self._all_symbols.append(sym)
        logger.info(f"Loaded {len(self._all_symbols)} USDT perp symbols")

    # ─── Main loop ────────────────────────────────────────────────────

    def _main_loop(self):
        """Main scanning loop."""
        listing_check_last = 0

        while self._running:
            try:
                loop_start = time.time()

                # 0. Reset daily PnL at midnight UTC (Fix #9)
                self._maybe_reset_daily_pnl()

                # 1. Check new listings periodically
                if time.time() - listing_check_last > LISTING_CHECK_INTERVAL:
                    self._check_new_listings()
                    listing_check_last = time.time()

                # 1b. Handle pending listing (non-blocking, Fix #4)
                if self._pending_listing and time.time() >= self._pending_listing["time"]:
                    coin = self._pending_listing["coin"]
                    self._pending_listing = None
                    self._execute_listing_buy(coin)

                # 2. Fetch all tickers in one call (efficient)
                tickers = self._fetch_all_tickers()
                if not tickers:
                    time.sleep(SCAN_INTERVAL_SEC)
                    continue

                # 3. Scan for pump signals
                signals = self._scan_for_pumps(tickers)

                # 4. Execute on confirmed signals
                for signal in signals:
                    self._execute_signal(signal)

                # 5. Manage open pump positions
                self._manage_positions(tickers)

                # 6. Sleep remaining interval
                elapsed = time.time() - loop_start
                sleep_time = max(0.1, SCAN_INTERVAL_SEC - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"PumpScanner main loop error: {e}", exc_info=True)
                time.sleep(5)

    def _maybe_reset_daily_pnl(self):
        """Reset daily PnL at midnight UTC (Fix #9)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_pump_pnl_date != today:
            if self._daily_pump_pnl_date:
                logger.info(f"Daily pump PnL reset (was ${self._daily_pump_pnl:.2f})")
            self._daily_pump_pnl = 0.0
            self._daily_pump_pnl_date = today

    # ═══════════════════════════════════════════════════════════════════
    # A. PUMP DETECTION
    # ═══════════════════════════════════════════════════════════════════

    def _fetch_all_tickers(self) -> dict:
        """Fetch tickers for all symbols in one API call."""
        try:
            tickers = self.client.fetch_tickers()
            return tickers
        except Exception as e:
            logger.error(f"Failed to fetch tickers: {e}")
            return {}

    def _scan_for_pumps(self, tickers: dict) -> list[PumpSignal]:
        """
        Scan all tickers for pump anomalies.

        Detection algorithm (Fix #6 — per-candle volume detection):
        1. Pre-filter: only coins with >10% 24h change from ticker
        2. For those coins, fetch 1m OHLCV (20 candles)
        3. Compare LAST candle volume to average of previous 19
        4. If volume_ratio > VOLUME_SPIKE_MULT AND price_change > PRICE_SPIKE_PCT:
           -> potential pump detected
        5. Validate with RSI, consecutive candles, buy ratio
        """
        signals: list[PumpSignal] = []
        now = time.time()

        for symbol, ticker in tickers.items():
            # Only USDT linear perps
            if symbol not in self._all_symbols:
                continue

            coin = symbol.split("/")[0]

            # Skip if on cooldown
            with self._lock:
                if coin in self.cooldowns and now < self.cooldowns[coin]:
                    continue
                # Skip if already in a pump position
                if coin in self.pump_positions:
                    continue

            try:
                last_price = float(ticker.get("last", 0))
                quote_volume = float(ticker.get("quoteVolume", 0) or 0)
                change_pct = float(ticker.get("percentage", 0) or 0) / 100  # convert to decimal

                if last_price <= 0 or quote_volume < MIN_DOLLAR_VOLUME:
                    continue

                # Update price history
                self.price_history[coin].append(last_price)
                if len(self.price_history[coin]) > 100:
                    self.price_history[coin] = self.price_history[coin][-100:]

                # Fix #6: Only fetch candle data for coins showing >10% 24h change
                if abs(change_pct) < 0.10:
                    continue

                # Fetch 1m OHLCV for per-candle volume detection
                try:
                    candles = self._safe_fetch_ohlcv(symbol, "1m", limit=20)
                except Exception:
                    continue
                if not candles or len(candles) < 5:
                    continue

                candle_volumes = [c[5] for c in candles]
                last_candle_vol = candle_volumes[-1]
                prev_avg_vol = np.mean(candle_volumes[:-1]) if len(candle_volumes) > 1 else 1.0

                volume_ratio = last_candle_vol / max(prev_avg_vol, 1e-9)

                # Store per-candle volume baseline for reference
                with self._lock:
                    self.volume_baselines[coin] = candle_volumes

                # Price change over window
                prices = self.price_history[coin]
                if len(prices) >= PRICE_WINDOW_CANDLES:
                    price_change = (prices[-1] - prices[-PRICE_WINDOW_CANDLES]) / prices[-PRICE_WINDOW_CANDLES]
                else:
                    price_change = change_pct

                # ── PRIMARY DETECTION: volume spike + price spike ──
                if volume_ratio >= VOLUME_SPIKE_MULT and price_change >= PRICE_SPIKE_PCT:
                    # Validate with RSI, consecutive candles, buy ratio
                    signal = self._validate_pump(coin, symbol, last_price, volume_ratio, price_change)
                    if signal:
                        signals.append(signal)

                # ── DUMP SHORT DETECTION: after a pump, detect exhaustion ──
                elif (volume_ratio >= VOLUME_SPIKE_MULT * 0.5 and
                      price_change >= PRICE_SPIKE_PCT * 2 and
                      len(prices) >= 20):
                    signal = self._check_dump_short(coin, symbol, last_price, volume_ratio)
                    if signal:
                        signals.append(signal)

            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
                continue

        return signals

    def _validate_pump(self, coin: str, symbol: str, price: float,
                       volume_ratio: float, price_change: float) -> Optional[PumpSignal]:
        """
        Validate a potential pump signal using 1m candle data.

        Checks:
        1. Consecutive green candles (CONFIRM_CANDLES)
        2. RSI in the sweet spot (60-85)
        3. Buy-side volume dominance
        4. ATR for stop-loss calculation
        """
        try:
            candles = self._safe_fetch_ohlcv(symbol, "1m", limit=30)
            if not candles or len(candles) < 15:
                return None

            closes = [c[4] for c in candles]
            opens = [c[1] for c in candles]
            highs = [c[2] for c in candles]
            lows = [c[3] for c in candles]
            volumes = [c[5] for c in candles]

            # 1. Consecutive green candles check
            green_count = 0
            for i in range(-1, -CONFIRM_CANDLES - 1, -1):
                if closes[i] > opens[i]:
                    green_count += 1
            if green_count < CONFIRM_CANDLES:
                return None

            # 2. RSI calculation (14-period)
            rsi = self._calc_rsi(closes, period=14)
            if rsi < MIN_RSI_ENTRY or rsi > MAX_RSI_ENTRY:
                return None

            # 3. Buy ratio: fraction of volume on green candles in last 5
            green_vol = sum(volumes[i] for i in range(-5, 0) if closes[i] > opens[i])
            total_vol = sum(volumes[-5:])
            buy_ratio = green_vol / max(total_vol, 1e-9)
            if buy_ratio < MIN_BUY_RATIO:
                return None

            # 4. ATR calculation (14-period)
            atr = self._calc_atr(highs, lows, closes, period=14)

            # 5. Confidence scoring (0-1)
            conf_vol = min(volume_ratio / (VOLUME_SPIKE_MULT * 2), 1.0)   # higher vol = better
            conf_price = min(price_change / (PRICE_SPIKE_PCT * 3), 1.0)   # bigger move = better
            conf_rsi = 1.0 - abs(rsi - 70) / 30                          # sweet spot around 70
            conf_buy = min(buy_ratio / 0.8, 1.0)                          # higher buy ratio = better
            confidence = (conf_vol * 0.3 + conf_price * 0.3 +
                         conf_rsi * 0.2 + conf_buy * 0.2)

            if confidence < 0.5:
                return None

            logger.info(
                f"PUMP DETECTED: {coin} | vol_ratio={volume_ratio:.1f}x | "
                f"price_change={price_change*100:.1f}% | RSI={rsi:.0f} | "
                f"buy_ratio={buy_ratio:.0%} | confidence={confidence:.2f}"
            )

            return PumpSignal(
                coin=coin,
                signal_type="pump_long",
                detected_at=time.time(),
                price_at_detection=price,
                volume_ratio=volume_ratio,
                rsi=rsi,
                atr=atr,
                confidence=confidence,
                metadata={
                    "price_change": price_change,
                    "buy_ratio": buy_ratio,
                    "green_candles": green_count,
                },
            )

        except Exception as e:
            logger.error(f"Pump validation failed for {coin}: {e}")
            return None

    def _check_dump_short(self, coin: str, symbol: str, price: float,
                          volume_ratio: float) -> Optional[PumpSignal]:
        """
        Detect pump exhaustion for a short entry.

        Exhaustion signals:
        1. RSI > 80 (overbought)
        2. Volume declining from peak (bearish divergence)
        3. Funding rate extreme (> 0.1%)
        4. Long upper wicks on recent candles (rejection)
        """
        try:
            candles = self._safe_fetch_ohlcv(symbol, "1m", limit=30)
            if not candles or len(candles) < 20:
                return None

            closes = [c[4] for c in candles]
            opens = [c[1] for c in candles]
            highs = [c[2] for c in candles]
            lows = [c[3] for c in candles]
            volumes = [c[5] for c in candles]

            # 1. RSI must be overbought
            rsi = self._calc_rsi(closes, period=14)
            if rsi < SHORT_RSI_THRESHOLD:
                return None

            # 2. Volume declining: compare last 5 candles avg vs peak 5 candles
            recent_vol = np.mean(volumes[-5:])
            peak_vol = np.max([np.mean(volumes[i:i+5]) for i in range(len(volumes)-10, len(volumes)-5)])
            vol_decline = 1 - (recent_vol / max(peak_vol, 1e-9))
            if vol_decline < SHORT_VOL_DECLINE_PCT:
                return None

            # 3. Check funding rate
            try:
                funding_data = self.client.fetch_funding_rate(symbol)
                funding = float(funding_data.get("fundingRate", 0))
            except Exception:
                funding = 0

            # 4. Upper wick ratio on last 3 candles (rejection signal)
            wick_scores = []
            for i in range(-3, 0):
                body = abs(closes[i] - opens[i])
                upper_wick = highs[i] - max(closes[i], opens[i])
                total_range = highs[i] - lows[i]
                if total_range > 0:
                    wick_scores.append(upper_wick / total_range)
            avg_wick = np.mean(wick_scores) if wick_scores else 0

            # Composite exhaustion score
            score_rsi = min((rsi - 75) / 20, 1.0)
            score_vol = min(vol_decline / 0.6, 1.0)
            score_funding = min(abs(funding) / SHORT_FUNDING_EXTREME, 1.0) if funding > 0 else 0
            score_wick = min(avg_wick / 0.5, 1.0)

            confidence = (score_rsi * 0.3 + score_vol * 0.3 +
                         score_funding * 0.2 + score_wick * 0.2)

            if confidence < 0.5:
                return None

            atr = self._calc_atr(highs, lows, closes, period=14)

            logger.info(
                f"DUMP SHORT SIGNAL: {coin} | RSI={rsi:.0f} | "
                f"vol_decline={vol_decline:.0%} | funding={funding:.4%} | "
                f"wick_ratio={avg_wick:.2f} | confidence={confidence:.2f}"
            )

            return PumpSignal(
                coin=coin,
                signal_type="dump_short",
                detected_at=time.time(),
                price_at_detection=price,
                volume_ratio=volume_ratio,
                rsi=rsi,
                atr=atr,
                confidence=confidence,
                metadata={
                    "vol_decline": vol_decline,
                    "funding": funding,
                    "avg_wick": avg_wick,
                },
            )

        except Exception as e:
            logger.error(f"Dump short check failed for {coin}: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════
    # B. TRADE EXECUTION
    # ═══════════════════════════════════════════════════════════════════

    def _execute_signal(self, signal: PumpSignal):
        """Execute a pump/dump/listing signal."""
        # Check budget
        with self._lock:
            if len(self.pump_positions) >= PUMP_MAX_POSITIONS:
                logger.info(f"Max pump positions reached, skipping {signal.coin}")
                return

        # Fix #13: Daily loss circuit breaker
        if self._daily_pump_pnl <= MAX_DAILY_PUMP_LOSS:
            logger.warning(f"Daily pump loss limit reached (${self._daily_pump_pnl:.2f}), skipping {signal.coin}")
            return

        # Verify coin has a tradable linear perpetual contract on Bybit FIRST
        symbol = f"{signal.coin}/USDT:USDT"
        markets = self.client.markets or {}
        if symbol not in markets:
            try:
                self.client.load_markets(True)
                markets = self.client.markets or {}
            except Exception:
                pass
        market_info = markets.get(symbol, {})
        is_tradable = (
            symbol in markets
            and market_info.get("active", False)
            and market_info.get("linear", False)
        )
        if not is_tradable:
            # Try a test fetch to be sure
            try:
                self.client.fetch_ticker(symbol)
            except Exception:
                logger.info(f"No tradable perpetual for {signal.coin}, skipping (24h cooldown)")
                with self._lock:
                    self.cooldowns[signal.coin] = time.time() + 86400
                return

        # Fix #18: Position conflict with main bot
        if self._main_bot_has_position(signal.coin):
            logger.info(f"Main bot already has position on {signal.coin}, skipping")
            return

        try:
            equity = self._get_equity()
            if equity <= 0:
                return

            if signal.signal_type == "pump_long":
                self._open_pump_long(signal, equity)
            elif signal.signal_type == "dump_short":
                self._open_dump_short(signal, equity)
            elif signal.signal_type == "new_listing":
                self._open_listing_long(signal, equity)

        except Exception as e:
            logger.error(f"Failed to execute signal for {signal.coin}: {e}")
            # Set cooldown on failure to stop retrying every 3 seconds
            with self._lock:
                if "not supported" in str(e) or "not allowed" in str(e):
                    self.cooldowns[signal.coin] = time.time() + 86400  # 24h for unsupported symbols
                else:
                    self.cooldowns[signal.coin] = time.time() + 300  # 5min for other errors

    def _main_bot_has_position(self, coin: str) -> bool:
        """Fix #18: Check if the main bot already has a position on this coin."""
        try:
            # Check exchange positions directly
            symbol = f"{coin}/USDT:USDT"
            positions = self.client.fetch_positions([symbol])
            for p in positions:
                contracts = float(p.get("contracts", 0) or 0)
                if contracts > 0 and p.get("symbol") == symbol:
                    # Position exists — could be main bot's
                    with self._lock:
                        if coin not in self.pump_positions:
                            # Not ours, must be main bot's
                            return True
        except Exception as e:
            logger.debug(f"Position conflict check failed for {coin}: {e}")
        return False

    def _open_pump_long(self, signal: PumpSignal, equity: float):
        """Open a long position to ride the pump."""
        coin = signal.coin
        symbol = f"{coin}/USDT:USDT"
        price = signal.price_at_detection
        atr = signal.atr

        # Position sizing: scale with confidence, capped at PUMP_RISK_BUDGET_PCT
        risk_frac = PUMP_RISK_BUDGET_PCT * signal.confidence
        notional = equity * risk_frac * PUMP_LEVERAGE
        quantity = notional / price

        # Round quantity to exchange precision
        quantity = self._round_qty(symbol, quantity)
        if quantity <= 0:
            return

        # Fix #10: Minimum notional check
        if notional < MIN_NOTIONAL:
            logger.info(f"Notional ${notional:.2f} below minimum ${MIN_NOTIONAL}, skipping {coin}")
            return

        # Set leverage
        try:
            self.client.set_leverage(PUMP_LEVERAGE, symbol)
        except Exception:
            pass  # may already be set

        # Place market buy
        order = self.client.create_market_order(symbol, "buy", quantity)
        fill_price = float(order.get("average", price) or price)

        # Calculate SL/TP levels
        sl = fill_price - (atr * PUMP_SL_ATR_MULT)
        tp1 = fill_price * (1 + PUMP_TP1_PCT)
        tp2 = fill_price * (1 + PUMP_TP2_PCT)
        tp3 = fill_price * (1 + PUMP_TP3_PCT)

        pos = PumpPosition(
            coin=coin, side="long",
            entry_price=fill_price, quantity=quantity,
            original_quantity=quantity,
            stop_loss=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            opened_at=time.time(),
        )
        with self._lock:
            self.pump_positions[coin] = pos

        msg = (
            f"PUMP LONG OPENED: {coin}\n"
            f"Entry: ${fill_price:.4f} | Qty: {quantity}\n"
            f"SL: ${sl:.4f} | TP1: ${tp1:.4f} | TP2: ${tp2:.4f} | TP3: ${tp3:.4f}\n"
            f"Vol ratio: {signal.volume_ratio:.1f}x | RSI: {signal.rsi:.0f} | "
            f"Conf: {signal.confidence:.0%}\n"
            f"Notional: ${notional:.0f} | Leverage: {PUMP_LEVERAGE}x"
        )
        logger.info(msg)
        self._alert(msg)

    def _open_dump_short(self, signal: PumpSignal, equity: float):
        """Open a short position after pump exhaustion."""
        coin = signal.coin
        symbol = f"{coin}/USDT:USDT"
        price = signal.price_at_detection
        atr = signal.atr

        risk_frac = PUMP_RISK_BUDGET_PCT * signal.confidence * 0.7  # smaller size for shorts
        notional = equity * risk_frac * PUMP_LEVERAGE
        quantity = notional / price
        quantity = self._round_qty(symbol, quantity)
        if quantity <= 0:
            return

        # Fix #10: Minimum notional check
        if notional < MIN_NOTIONAL:
            logger.info(f"Notional ${notional:.2f} below minimum ${MIN_NOTIONAL}, skipping short {coin}")
            return

        try:
            self.client.set_leverage(PUMP_LEVERAGE, symbol)
        except Exception:
            pass

        order = self.client.create_market_order(symbol, "sell", quantity)
        fill_price = float(order.get("average", price) or price)

        sl = fill_price + (atr * SHORT_SL_ATR_MULT)
        tp1 = fill_price * (1 - SHORT_TP_PCT * 0.5)
        tp2 = fill_price * (1 - SHORT_TP_PCT)
        tp3 = fill_price * (1 - SHORT_TP_PCT * 1.5)

        pos = PumpPosition(
            coin=coin, side="short",
            entry_price=fill_price, quantity=quantity,
            original_quantity=quantity,
            stop_loss=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            opened_at=time.time(),
        )
        with self._lock:
            self.pump_positions[coin] = pos

        msg = (
            f"DUMP SHORT OPENED: {coin}\n"
            f"Entry: ${fill_price:.4f} | Qty: {quantity}\n"
            f"SL: ${sl:.4f} | TP: ${tp2:.4f}\n"
            f"RSI: {signal.rsi:.0f} | Vol decline: {signal.metadata.get('vol_decline', 0):.0%}\n"
            f"Funding: {signal.metadata.get('funding', 0):.4%}"
        )
        logger.info(msg)
        self._alert(msg)

    def _open_listing_long(self, signal: PumpSignal, equity: float):
        """Open a long position on a newly listed coin."""
        coin = signal.coin
        symbol = f"{coin}/USDT:USDT"
        price = signal.price_at_detection

        notional = equity * LISTING_RISK_PCT * PUMP_LEVERAGE
        quantity = notional / price
        quantity = self._round_qty(symbol, quantity)
        if quantity <= 0:
            return

        # Fix #10: Minimum notional check
        if notional < MIN_NOTIONAL:
            logger.info(f"Notional ${notional:.2f} below minimum ${MIN_NOTIONAL}, skipping listing {coin}")
            return

        try:
            self.client.set_leverage(PUMP_LEVERAGE, symbol)
        except Exception:
            pass

        order = self.client.create_market_order(symbol, "buy", quantity)
        fill_price = float(order.get("average", price) or price)

        # New listings: tight SL (5%), aggressive TP
        sl = fill_price * 0.95
        tp1 = fill_price * 1.10
        tp2 = fill_price * 1.25
        tp3 = fill_price * 1.50

        pos = PumpPosition(
            coin=coin, side="long",
            entry_price=fill_price, quantity=quantity,
            original_quantity=quantity,
            stop_loss=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            opened_at=time.time(),
        )
        with self._lock:
            self.pump_positions[coin] = pos

        msg = (
            f"NEW LISTING LONG: {coin}\n"
            f"Entry: ${fill_price:.4f} | Qty: {quantity}\n"
            f"SL: ${sl:.4f} (-5%) | TP1: ${tp1:.4f} (+10%)"
        )
        logger.info(msg)
        self._alert(msg)

    # ═══════════════════════════════════════════════════════════════════
    # C. POSITION MANAGEMENT — Partial TP + Trailing Stop
    # ═══════════════════════════════════════════════════════════════════

    def _manage_positions(self, tickers: dict):
        """Check all pump positions for SL/TP/trailing."""
        with self._lock:
            positions_snapshot = list(self.pump_positions.items())

        for coin, pos in positions_snapshot:
            symbol = f"{coin}/USDT:USDT"
            ticker = tickers.get(symbol)
            if not ticker:
                continue

            current_price = float(ticker.get("last", 0))
            if current_price <= 0:
                continue

            is_long = pos.side == "long"

            # ── STOP LOSS ──
            if is_long and current_price <= pos.stop_loss:
                self._close_pump_position(coin, "SL HIT", current_price)
                continue
            elif not is_long and current_price >= pos.stop_loss:
                self._close_pump_position(coin, "SL HIT", current_price)
                continue

            # ── TIME-BASED EXIT: close after 2 hours max ──
            if time.time() - pos.opened_at > 7200:
                self._close_pump_position(coin, "TIME EXIT (2h)", current_price)
                continue

            # ── PARTIAL TAKE PROFITS (long) — Fix #2: separate if blocks ──
            if is_long:
                if not pos.tp1_hit and current_price >= pos.tp1:
                    # TP1: close 40% of position
                    close_qty = self._round_qty(symbol, pos.original_quantity * 0.4)
                    if close_qty > 0:
                        self._partial_close(coin, symbol, close_qty, "TP1 (+5%)")
                        pos.quantity = max(0, pos.quantity - close_qty)  # Fix #3
                        pos.tp1_hit = True
                        # Move SL to breakeven
                        pos.stop_loss = pos.entry_price * 1.002  # slight profit lock

                if not pos.tp2_hit and pos.tp1_hit and current_price >= pos.tp2:
                    # TP2: close 30% of position, activate trailing
                    close_qty = self._round_qty(symbol, pos.original_quantity * 0.3)
                    if close_qty > 0:
                        self._partial_close(coin, symbol, close_qty, "TP2 (+10%)")
                        pos.quantity = max(0, pos.quantity - close_qty)  # Fix #3
                        pos.tp2_hit = True
                        pos.trailing_active = True
                        pos.trailing_high = current_price

                if not pos.tp3_hit and pos.tp2_hit and current_price >= pos.tp3:
                    # TP3: close remaining
                    pos.tp3_hit = True
                    self._close_pump_position(coin, "TP3 (+20%)", current_price)
                    continue

                # Trailing stop after TP2
                if pos.trailing_active:
                    if current_price > pos.trailing_high:
                        pos.trailing_high = current_price
                    trailing_sl = pos.trailing_high * (1 - PUMP_TRAILING_PCT)
                    if current_price <= trailing_sl:
                        self._close_pump_position(coin, "TRAILING STOP", current_price)
                        continue

            # ── PARTIAL TAKE PROFITS (short) — Fix #2: separate if blocks ──
            else:
                if not pos.tp1_hit and current_price <= pos.tp1:
                    close_qty = self._round_qty(symbol, pos.original_quantity * 0.4)
                    if close_qty > 0:
                        self._partial_close_short(coin, symbol, close_qty, "TP1")
                        pos.quantity = max(0, pos.quantity - close_qty)  # Fix #3
                        pos.tp1_hit = True
                        pos.stop_loss = pos.entry_price * 0.998

                if not pos.tp2_hit and pos.tp1_hit and current_price <= pos.tp2:
                    close_qty = self._round_qty(symbol, pos.original_quantity * 0.3)
                    if close_qty > 0:
                        self._partial_close_short(coin, symbol, close_qty, "TP2")
                        pos.quantity = max(0, pos.quantity - close_qty)  # Fix #3
                        pos.tp2_hit = True
                        pos.trailing_active = True
                        pos.trailing_high = current_price  # actually trailing low

                if not pos.tp3_hit and pos.tp2_hit and current_price <= pos.tp3:
                    pos.tp3_hit = True
                    self._close_pump_position(coin, "TP3", current_price)
                    continue

                if pos.trailing_active:
                    if current_price < pos.trailing_high:
                        pos.trailing_high = current_price
                    trailing_sl = pos.trailing_high * (1 + PUMP_TRAILING_PCT)
                    if current_price >= trailing_sl:
                        self._close_pump_position(coin, "TRAILING STOP", current_price)
                        continue

    def _partial_close(self, coin: str, symbol: str, qty: float, reason: str):
        """Partially close a long position."""
        try:
            self.client.create_market_order(symbol, "sell", qty)
            logger.info(f"PARTIAL CLOSE {coin} {reason}: sold {qty}")
            self._alert(f"PUMP {coin} {reason}: partial close {qty}")
        except Exception as e:
            logger.error(f"Partial close failed for {coin}: {e}")

    def _partial_close_short(self, coin: str, symbol: str, qty: float, reason: str):
        """Partially close a short position."""
        try:
            self.client.create_market_order(symbol, "buy", qty)
            logger.info(f"PARTIAL CLOSE SHORT {coin} {reason}: bought {qty}")
            self._alert(f"DUMP SHORT {coin} {reason}: partial close {qty}")
        except Exception as e:
            logger.error(f"Partial close short failed for {coin}: {e}")

    def _close_pump_position(self, coin: str, reason: str, exit_price: float):
        """Fully close a pump position."""
        with self._lock:
            pos = self.pump_positions.get(coin)
        if not pos:
            return

        symbol = f"{coin}/USDT:USDT"
        try:
            side = "sell" if pos.side == "long" else "buy"
            remaining_qty = self._round_qty(symbol, pos.quantity)
            if remaining_qty > 0:
                self.client.create_market_order(symbol, side, remaining_qty)

            # Fix #1: Use pos.quantity (current remaining) not original_quantity for PnL
            if pos.side == "long":
                pnl = (exit_price - pos.entry_price) * pos.quantity
            else:
                pnl = (pos.entry_price - exit_price) * pos.quantity

            pnl_pct = (exit_price / pos.entry_price - 1) * 100
            if pos.side == "short":
                pnl_pct = -pnl_pct

            self._daily_pump_pnl += pnl
            duration = time.time() - pos.opened_at

            msg = (
                f"PUMP CLOSED: {coin} ({pos.side}) | {reason}\n"
                f"Entry: ${pos.entry_price:.4f} -> Exit: ${exit_price:.4f}\n"
                f"PnL: ${pnl:.2f} ({pnl_pct:+.1f}%) | Duration: {duration/60:.0f}min\n"
                f"Daily pump PnL: ${self._daily_pump_pnl:.2f}"
            )
            logger.info(msg)
            self._alert(msg)

            # Save trade to file for verification
            try:
                import json as _json
                trade_record = {
                    "coin": coin, "side": pos.side,
                    "entry_price": pos.entry_price, "exit_price": exit_price,
                    "quantity": pos.original_quantity, "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 1), "reason": reason,
                    "duration_min": round(duration / 60, 1),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                pump_log_path = os.path.join(os.path.dirname(__file__), "pump_trades.json")
                try:
                    with open(pump_log_path, "r") as f:
                        pump_log = _json.load(f)
                except Exception:
                    pump_log = []
                pump_log.append(trade_record)
                if len(pump_log) > 200:
                    pump_log = pump_log[-200:]
                with open(pump_log_path, "w") as f:
                    _json.dump(pump_log, f, indent=2)
            except Exception as _le:
                logger.debug(f"Pump trade log save failed: {_le}")

        except Exception as e:
            logger.error(f"Close pump position failed for {coin}: {e}")

        # Remove and set cooldown
        with self._lock:
            self.pump_positions.pop(coin, None)
            self.cooldowns[coin] = time.time() + PUMP_COOLDOWN_SEC

    # ═══════════════════════════════════════════════════════════════════
    # D. NEW LISTING DETECTION
    # ═══════════════════════════════════════════════════════════════════

    def _check_new_listings(self):
        """
        Check Bybit announcements API for new perpetual listings.

        Uses the official Bybit V5 API endpoint:
        GET https://api.bybit.com/v5/announcements/index
        """
        try:
            # Method 1: Official Bybit announcements API
            url = "https://api.bybit.com/v5/announcements/index"
            params = {
                "locale": "en-US",
                "type": "new_crypto",
                "limit": 10,
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("result", {}).get("list", [])
                for item in items:
                    title = item.get("title", "").upper()
                    desc = item.get("description", "").upper()

                    # Look for "USDT PERPETUAL" in announcement
                    if "PERPETUAL" in title or "PERP" in title:
                        # Extract coin symbol from title
                        # Pattern: "Bybit Lists XXXUSDT Perpetual Contract"
                        coin = self._extract_coin_from_listing(title)
                        if coin and coin not in self.known_listings:
                            self.known_listings.add(coin)
                            logger.info(f"NEW LISTING DETECTED: {coin}")
                            self._handle_new_listing(coin)

            # Method 2: Check for new symbols in market data
            self._check_new_symbols()

        except Exception as e:
            logger.error(f"Listing check failed: {e}")

    def _extract_coin_from_listing(self, title: str) -> Optional[str]:
        """Extract coin symbol from listing announcement title."""
        # "Bybit Lists XYZUSDT Perpetual Contract"
        import re
        match = re.search(r'([A-Z0-9]{2,10})USDT', title)
        if match:
            return match.group(1)
        return None

    def _check_new_symbols(self):
        """Check if any new USDT perp symbols appeared on the exchange."""
        try:
            # Fix #7: Cache markets for MARKET_CACHE_TTL_SEC instead of reloading every cycle
            now = time.time()
            if now - self._markets_last_loaded < MARKET_CACHE_TTL_SEC:
                return
            self.client.load_markets(True)  # force reload
            self._markets_last_loaded = now

            current_symbols = set()
            for sym, info in self.client.markets.items():
                if (info.get("linear") and info.get("active") and
                    info.get("quote") == "USDT" and info.get("type") == "swap"):
                    current_symbols.add(sym)

            # Find new symbols
            old_symbols = set(self._all_symbols)
            new_symbols = current_symbols - old_symbols

            for sym in new_symbols:
                coin = sym.split("/")[0]
                if coin not in self.known_listings:
                    self.known_listings.add(coin)
                    logger.info(f"NEW SYMBOL DETECTED: {coin} ({sym})")
                    self._handle_new_listing(coin)

            # Update symbol list
            self._all_symbols = list(current_symbols)

        except Exception as e:
            logger.debug(f"Symbol check failed: {e}")


    def _check_binance_announcements(self):
        """Check Binance for new listing announcements. If a coin lists on Binance, it often pumps on Bybit too."""
        try:
            import requests
            url = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
            params = {"type": 1, "catalogId": 48, "pageNo": 1, "pageSize": 5}
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                articles = data.get("data", {}).get("catalogs", [{}])[0].get("articles", [])
                for article in articles[:3]:
                    title = article.get("title", "").upper()
                    if "LIST" in title and ("PERPETUAL" in title or "FUTURES" in title):
                        # Extract coin symbol from title
                        import re
                        match = re.search(r"\b([A-Z]{2,10})USDT\b", title)
                        if match:
                            coin = match.group(1)
                            # Check if this coin exists on Bybit
                            sym = f"{coin}/USDT:USDT"
                            if sym in (self.client.markets or {}):
                                if self._tg:
                                    self._tg(f"BINANCE LISTING DETECTED: {coin} - also available on Bybit!")
                                logger.info(f"[LISTING] Binance listed {coin}, available on Bybit")
        except Exception as e:
            logger.debug(f"Binance announcement check failed: {e}")


    def _handle_new_listing(self, coin: str):
        """Handle a detected new listing (non-blocking, Fix #4)."""
        # Skip blacklisted stock tokens
        if coin in self._blacklist:
            logger.info(f"Skipping blacklisted stock token: {coin}")
            return
        # Skip already-seen listings
        if coin in self._seen_listings:
            return
        self._seen_listings.add(coin)
        # Persist seen listings
        try:
            import json as _json
            with open(self._seen_listings_file, "w") as f:
                _json.dump(list(self._seen_listings), f)
        except Exception:
            pass
        self._alert(f"NEW LISTING DETECTED: {coin} -- preparing to buy in {LISTING_BUY_DELAY_SEC}s")

        # Fix #4: Set pending listing flag instead of blocking with sleep
        self._pending_listing = {
            "coin": coin,
            "time": time.time() + LISTING_BUY_DELAY_SEC,
        }

    def _execute_listing_buy(self, coin: str):
        """Execute the actual listing buy after the delay has passed (Fix #4)."""
        symbol = f"{coin}/USDT:USDT"
        try:
            # Verify the pair exists and get current price
            ticker = self.client.fetch_ticker(symbol)
            price = float(ticker.get("last", 0))
            if price <= 0:
                logger.warning(f"No price for new listing {coin}")
                return

            signal = PumpSignal(
                coin=coin,
                signal_type="new_listing",
                detected_at=time.time(),
                price_at_detection=price,
                volume_ratio=0,
                rsi=50,
                atr=price * 0.02,  # estimate ATR as 2% of price
                confidence=0.7,
                metadata={"source": "listing_detection"},
            )
            self._execute_signal(signal)

        except Exception as e:
            logger.error(f"Failed to trade new listing {coin}: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # TECHNICAL INDICATORS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _calc_rsi(closes: list[float], period: int = 14) -> float:
        """Calculate RSI from a list of close prices."""
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calc_atr(highs: list[float], lows: list[float],
                  closes: list[float], period: int = 14) -> float:
        """Calculate ATR from OHLC data."""
        if len(highs) < period + 1:
            return 0.0
        trs = []
        for i in range(-period, 0):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        return np.mean(trs)

    # ═══════════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════════

    def _get_equity(self) -> float:
        """Get account equity."""
        try:
            bal = self.client.fetch_balance({"type": "contract"})
            return float(bal["total"].get("USDT", 0))
        except Exception:
            return 0.0

    def _round_qty(self, symbol: str, qty: float) -> float:
        """Round quantity to exchange-allowed precision."""
        try:
            market = self.client.market(symbol)
            precision = market.get("precision", {}).get("amount", 8)
            min_qty = market.get("limits", {}).get("amount", {}).get("min", 0)
            qty = float(self.client.amount_to_precision(symbol, qty))
            if qty < min_qty:
                return 0.0
            return qty
        except Exception:
            return round(qty, 4)

    def _alert(self, message: str):
        """Send Telegram alert."""
        if self.telegram_fn:
            try:
                self.telegram_fn(message)
            except Exception:
                pass
        elif PUMP_TELEGRAM_TOKEN and PUMP_TELEGRAM_CHAT_ID:
            try:
                url = f"https://api.telegram.org/bot{PUMP_TELEGRAM_TOKEN}/sendMessage"
                requests.post(url, json={
                    "chat_id": PUMP_TELEGRAM_CHAT_ID,
                    "text": f"[PUMP SCANNER]\n{message}",
                    "parse_mode": "HTML",
                }, timeout=5)
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════════
    # STATUS / DEBUGGING
    # ═══════════════════════════════════════════════════════════════════

    def get_status(self) -> dict:
        """Return current scanner status for dashboard/monitoring."""
        with self._lock:
            return {
                "running": self._running,
                "symbols_monitored": len(self._all_symbols),
                "active_pump_positions": len(self.pump_positions),
                "positions": {
                    coin: {
                        "side": pos.side,
                        "entry": pos.entry_price,
                        "sl": pos.stop_loss,
                        "tp1": pos.tp1,
                        "tp2": pos.tp2,
                        "trailing_active": pos.trailing_active,
                        "trailing_high": pos.trailing_high,
                    }
                    for coin, pos in self.pump_positions.items()
                },
                "cooldowns_active": sum(1 for t in self.cooldowns.values() if t > time.time()),
                "daily_pump_pnl": self._daily_pump_pnl,
                "volume_baselines_loaded": len(self.volume_baselines),
            }


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION HELPER — add to existing bot
# ═══════════════════════════════════════════════════════════════════════════

def create_pump_scanner_from_config():
    """
    Factory function to create a PumpScanner from environment config.

    Usage in pro_trader.py:
        from pump_scanner import create_pump_scanner_from_config
        scanner = create_pump_scanner_from_config()
        if scanner:
            scanner.start()
    """
    try:
        import ccxt

        api_key = os.getenv("BYBIT_API_KEY", "")
        api_secret = os.getenv("BYBIT_API_SECRET", "")

        if not api_key or not api_secret:
            logger.warning("Bybit credentials not set, pump scanner disabled")
            return None

        client = ccxt.bybit({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "linear"},
        })
        testnet = os.getenv("BYBIT_TESTNET", "false").lower() == "true"
        if testnet:
            client.set_sandbox_mode(True)
        client.load_markets()

        scanner = PumpScanner(client)
        return scanner

    except Exception as e:
        logger.error(f"Failed to create pump scanner: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE MODE — run directly: python pump_scanner.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    print("""
    ╔═══════════════════════════════════════════╗
    ║   DeepAlpha Pump Scanner                  ║
    ║   Real-time pump detection on Bybit       ║
    ║   https://deepalphabot.com                ║
    ╚═══════════════════════════════════════════╝
    """)

    scanner = create_pump_scanner_from_config()
    if scanner:
        print(f"Pump Scanner initialized. Starting...")
        scanner.start()
        # Keep main thread alive
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nShutting down...")
            scanner.stop()
    else:
        print("Failed to initialize. Check your .env file:")
        print("  BYBIT_API_KEY=your_key")
        print("  BYBIT_API_SECRET=your_secret")
