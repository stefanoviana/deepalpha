"""
DeepAlpha — Risk Manager (Free Version)
Handles position sizing, stop-loss/take-profit, daily loss tracking,
and circuit breaker logic.
"""

import time
import config


class RiskManager:
    """Enforces risk rules for every trade decision."""

    def __init__(self):
        self.daily_pnl: float = 0.0
        self.daily_reset_ts: float = time.time()
        self.consecutive_losses: int = 0
        self.circuit_breaker_until: float = 0.0
        self.open_positions: dict[str, dict] = {}  # coin -> position info

    # ─── Daily reset ────────────────────────────────────────────────────

    def _check_daily_reset(self) -> None:
        """Reset daily counters at midnight UTC."""
        now = time.time()
        if now - self.daily_reset_ts >= 86_400:
            self.daily_pnl = 0.0
            self.daily_reset_ts = now

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
            pnl = (exit_price - pos["entry"]) * pos["qty"]
        else:
            pnl = (pos["entry"] - exit_price) * pos["qty"]

        self.daily_pnl += pnl

        # Track consecutive losses for circuit breaker
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= config.CIRCUIT_BREAKER_LOSSES:
                self.circuit_breaker_until = time.time() + config.CIRCUIT_BREAKER_COOLDOWN
                self.consecutive_losses = 0
        else:
            self.consecutive_losses = 0

        return round(pnl, 2)

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
