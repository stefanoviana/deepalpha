"""
Tests for DeepAlpha risk manager (risk_manager.py).
"""

import time
import pytest

from risk_manager import RiskManager


class TestPositionSizing:
    """Tests for calc_position_size and calc_quantity."""

    def test_position_size_is_percentage_of_equity(self, mock_config):
        """Position size = equity * RISK_PER_TRADE * LEVERAGE."""
        rm = RiskManager()
        equity = 10000.0
        price = 50000.0
        expected = equity * 0.10 * 5  # 5000.0
        assert rm.calc_position_size(equity, price) == expected

    def test_position_size_scales_with_equity(self, mock_config):
        """Doubling equity should double position size."""
        rm = RiskManager()
        size_1k = rm.calc_position_size(1000.0, 50000.0)
        size_2k = rm.calc_position_size(2000.0, 50000.0)
        assert size_2k == pytest.approx(size_1k * 2)

    def test_calc_quantity(self, mock_config):
        """Quantity = notional / price."""
        rm = RiskManager()
        equity = 10000.0
        price = 50000.0
        qty = rm.calc_quantity(equity, price)
        notional = equity * 0.10 * 5
        assert qty == pytest.approx(notional / price)


class TestDailyLossLimit:
    """Tests for daily loss limit enforcement."""

    def test_can_open_within_daily_limit(self, mock_config):
        """Should allow trading when daily loss is within limit."""
        rm = RiskManager()
        allowed, reason = rm.can_open(10000.0)
        assert allowed is True
        assert reason == "OK"

    def test_blocked_when_daily_loss_exceeded(self, mock_config):
        """Should block trading when daily loss exceeds MAX_DAILY_LOSS_PCT."""
        rm = RiskManager()
        # Simulate daily loss of 6% on 10000 equity (-600)
        rm.daily_pnl = -600.0
        allowed, reason = rm.can_open(10000.0)
        assert allowed is False
        assert "Daily loss limit" in reason

    def test_exactly_at_daily_limit(self, mock_config):
        """Should block when daily P&L exactly equals the negative limit."""
        rm = RiskManager()
        # daily_pnl / equity = -0.05 exactly
        rm.daily_pnl = -500.0
        allowed, reason = rm.can_open(10000.0)
        assert allowed is False


class TestCircuitBreaker:
    """Tests for circuit breaker logic (consecutive losses trigger pause)."""

    def test_circuit_breaker_triggers_after_3_losses(self, mock_config):
        """3 consecutive losses should activate the circuit breaker."""
        rm = RiskManager()

        for i in range(3):
            rm.register_open(f"COIN{i}", "long", 100.0, 1.0)
            rm.register_close(f"COIN{i}", 90.0)  # loss

        allowed, reason = rm.can_open(10000.0)
        assert allowed is False
        assert "Circuit breaker" in reason

    def test_circuit_breaker_resets_on_win(self, mock_config):
        """A winning trade should reset the consecutive loss counter."""
        rm = RiskManager()

        # 2 losses
        rm.register_open("A", "long", 100.0, 1.0)
        rm.register_close("A", 90.0)
        rm.register_open("B", "long", 100.0, 1.0)
        rm.register_close("B", 90.0)

        assert rm.consecutive_losses == 2

        # 1 win resets counter
        rm.register_open("C", "long", 100.0, 1.0)
        rm.register_close("C", 110.0)

        assert rm.consecutive_losses == 0

    def test_circuit_breaker_cooldown_expires(self, mock_config, monkeypatch):
        """After cooldown, trading should be allowed again."""
        rm = RiskManager()

        # Trigger circuit breaker
        for i in range(3):
            rm.register_open(f"X{i}", "long", 100.0, 1.0)
            rm.register_close(f"X{i}", 90.0)

        # Fast-forward past cooldown
        rm.circuit_breaker_until = time.time() - 1

        allowed, reason = rm.can_open(10000.0)
        assert allowed is True


class TestMaxPositions:
    """Tests for max concurrent positions check."""

    def test_blocked_at_max_positions(self, mock_config):
        """Should block when MAX_POSITIONS open positions exist."""
        rm = RiskManager()
        for i in range(3):  # MAX_POSITIONS = 3
            rm.register_open(f"COIN{i}", "long", 100.0, 1.0)

        allowed, reason = rm.can_open(10000.0)
        assert allowed is False
        assert "Max positions" in reason

    def test_allowed_below_max_positions(self, mock_config):
        """Should allow when fewer than MAX_POSITIONS are open."""
        rm = RiskManager()
        rm.register_open("BTC", "long", 50000.0, 0.1)
        rm.register_open("ETH", "long", 3000.0, 1.0)

        allowed, reason = rm.can_open(10000.0)
        assert allowed is True

    def test_closing_position_frees_slot(self, mock_config):
        """Closing a position should free a slot for a new one."""
        rm = RiskManager()
        for i in range(3):
            rm.register_open(f"COIN{i}", "long", 100.0, 1.0)

        # Full, should be blocked
        allowed, _ = rm.can_open(10000.0)
        assert allowed is False

        # Close one
        rm.register_close("COIN0", 105.0)

        allowed, reason = rm.can_open(10000.0)
        assert allowed is True


class TestSLTP:
    """Tests for stop-loss and take-profit calculation."""

    def test_long_sl_tp(self, mock_config):
        """Long SL should be below entry, TP above."""
        rm = RiskManager()
        sl, tp = rm.calc_sl_tp(100.0, "long")
        assert sl < 100.0
        assert tp > 100.0

    def test_short_sl_tp(self, mock_config):
        """Short SL should be above entry, TP below."""
        rm = RiskManager()
        sl, tp = rm.calc_sl_tp(100.0, "short")
        assert sl > 100.0
        assert tp < 100.0
