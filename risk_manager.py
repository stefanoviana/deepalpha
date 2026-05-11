"""
DeepAlpha — Risk Manager (Free Version)
Handles position sizing, stop-loss/take-profit, daily loss tracking,
and circuit breaker logic.
"""

import time
from datetime import datetime, timezone
from collections import deque

import config
from config import FEE_RATE


class RiskManager:
    """Enforces risk rules for every trade decision."""

    def __init__(self):
        self.daily_pnl: float = 0.0
        self._last_reset_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.consecutive_losses: int = 0
        self.circuit_breaker_until: float = 0.0
        self.open_positions: dict[str, dict] = {}  # coin -> position info
        self.closed_trades: deque[dict] = deque(maxlen=100)

    # ─── Daily reset ────────────────────────────────────────────────────

    def _check_daily_reset(self) -> None:
        """Reset daily counters at UTC midnight."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._last_reset_date:
            self.daily_pnl = 0.0
            self._last_reset_date = today

    # ─── Can we open a new trade? ───────────────────────────────────────

    def can_open(self, equity: float) -> tuple[bool, str]:
        """
        Check whether opening a new position is allowed.

        Returns
        -------
        (allowed, reason)
        """
        self._check_daily_reset()

        # Circuit breaker active?
        if time.time() < self.circuit_breaker_until:
            remaining = int(self.circuit_breaker_until - time.time())
            return False, f"Circuit breaker active ({remaining}s left)"

        # Max positions reached?
        if len(self.open_positions) >= config.MAX_POSITIONS:
            return False, f"Max positions reached ({config.MAX_POSITIONS})"

        # Daily loss limit hit?
        if equity > 0 and (self.daily_pnl / equity) <= -config.MAX_DAILY_LOSS_PCT:
            return False, f"Daily loss limit hit ({config.MAX_DAILY_LOSS_PCT*100:.1f}%)"

        return True, "OK"

    # ─── Position sizing ────────────────────────────────────────────────

    def calc_position_size(self, equity: float, price: float) -> float:
        """
        Calculate the notional position size in USD.

        Uses RISK_PER_TRADE * equity * LEVERAGE.
        """
        notional = equity * config.RISK_PER_TRADE * config.LEVERAGE
        return round(notional, 2)

    def calc_quantity(self, equity: float, price: float) -> float:
        """Calculate the asset quantity for the trade."""
        notional = self.calc_position_size(equity, price)
        qty = notional / price
        return qty

    # ─── Stop-loss & take-profit ────────────────────────────────────────

    def calc_sl_tp(self, entry_price: float, side: str) -> tuple[float, float]:
        """
        Calculate fixed stop-loss and take-profit prices.

        Parameters
        ----------
        entry_price : float
        side : str — "long" or "short"

        Returns
        -------
        (stop_loss_price, take_profit_price)
        """
        if side == "long":
            sl = entry_price * (1 - config.STOP_LOSS_PCT)
            tp = entry_price * (1 + config.TAKE_PROFIT_PCT)
        else:
            sl = entry_price * (1 + config.STOP_LOSS_PCT)
            tp = entry_price * (1 - config.TAKE_PROFIT_PCT)
        return round(sl, 6), round(tp, 6)

    # ─── Position tracking ──────────────────────────────────────────────

    def register_open(self, coin: str, side: str, entry_price: float, qty: float) -> None:
        """Record a new open position."""
        sl, tp = self.calc_sl_tp(entry_price, side)
        self.open_positions[coin] = {
            "side": side,
            "entry": entry_price,
            "qty": qty,
            "sl": sl,
            "tp": tp,
            "opened_at": time.time(),
        }

    def register_close(self, coin: str, exit_price: float) -> float:
        """
        Record a position close and update P&L tracking.

        Returns the realised P&L in USD.
        """
        if coin not in self.open_positions:
            return 0.0

        pos = self.open_positions.pop(coin)
        if pos["side"] == "long":
            raw_pnl = (exit_price - pos["entry"]) * pos["qty"]
        else:
            raw_pnl = (pos["entry"] - exit_price) * pos["qty"]

        # Subtract fees on both entry and exit (2 sides)
        notional_entry = pos["entry"] * pos["qty"]
        notional_exit = exit_price * pos["qty"]
        fees = FEE_RATE * notional_entry + FEE_RATE * notional_exit
        pnl = raw_pnl - fees

        self.daily_pnl += pnl
        rounded_pnl = round(pnl, 2)
        self.closed_trades.append(
            {
                "coin": coin,
                "side": pos["side"],
                "entry": pos["entry"],
                "exit": exit_price,
                "qty": pos["qty"],
                "pnl": rounded_pnl,
                "closed_at": time.time(),
            }
        )

        # Track consecutive losses for circuit breaker
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= config.CIRCUIT_BREAKER_LOSSES:
                self.circuit_breaker_until = time.time() + config.CIRCUIT_BREAKER_COOLDOWN
                self.consecutive_losses = 0
        else:
            self.consecutive_losses = 0

        return rounded_pnl

    def get_signal_performance(self, recent: int = 30) -> dict:
        """
        Return signal performance metrics for Telegram status surfaces.

        Metrics are based on realised trades tracked by this process. They are
        intentionally side-effect free so dashboards, CLI status, and Telegram
        commands can call this without touching exchange state.
        """
        trades = list(self.closed_trades)[-recent:] if recent > 0 else list(self.closed_trades)
        total = len(trades)
        wins = [trade for trade in trades if trade["pnl"] > 0]
        losses = [trade for trade in trades if trade["pnl"] < 0]
        total_pnl = sum(trade["pnl"] for trade in trades)

        return {
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / total * 100) if total else 0.0,
            "average_pnl": (total_pnl / total) if total else 0.0,
            "total_pnl": total_pnl,
        }

    def format_signal_performance(self, recent: int = 30) -> str:
        """Render signal performance for human-facing Telegram/status messages."""
        metrics = self.get_signal_performance(recent=recent)
        return (
            f"Signals ({metrics['total_trades']} closed): "
            f"Win rate {metrics['win_rate']:.1f}% | "
            f"Avg PnL ${metrics['average_pnl']:.2f} | "
            f"Total PnL ${metrics['total_pnl']:.2f}"
        )

    # ─── SL/TP check ───────────────────────────────────────────────────

    def check_exit(self, coin: str, current_price: float) -> str | None:
        """
        Check if a position should be closed due to SL or TP.

        Returns "sl", "tp", or None.
        """
        if coin not in self.open_positions:
            return None

        pos = self.open_positions[coin]

        if pos["side"] == "long":
            if current_price <= pos["sl"]:
                return "sl"
            if current_price >= pos["tp"]:
                return "tp"
        else:
            if current_price >= pos["sl"]:
                return "sl"
            if current_price <= pos["tp"]:
                return "tp"

        return None
