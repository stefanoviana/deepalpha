"""Tests for the DeepAlpha Telegram bot."""

import json
import pytest
from unittest.mock import patch

from telegram_bot import (
    format_signal,
    send_telegram_message,
    send_signal,
    SIGNAL_TEMPLATE,
)


def test_format_signal_long():
    """Test formatting a LONG signal."""
    signal = {
        'direction': 'LONG',
        'pair': 'BTC/USDT',
        'confidence': 87,
        'entry': 84500,
        'take_profit': 92000,
        'stop_loss': 80000,
        'reasoning': 'Strong bullish divergence detected.',
    }
    result = format_signal(signal)
    assert 'LONG' in result
    assert 'BTC/USDT' in result
    assert '87' in result
    assert '84500' in result
    assert '92000' in result
    assert '80000' in result
    assert 'Strong bullish divergence' in result


def test_format_signal_short():
    """Test formatting a SHORT signal."""
    signal = {
        'direction': 'SHORT',
        'pair': 'ETH/USDT',
        'confidence': 72,
        'entry': 3200,
        'take_profit': 2800,
        'stop_loss': 3500,
        'reasoning': 'Bearish flag pattern on 1H.',
    }
    result = format_signal(signal)
    assert 'SHORT' in result
    assert 'ETH/USDT' in result
    assert '72' in result


def test_format_signal_missing_fields():
    """Test formatting handles missing fields gracefully."""
    signal = {
        'direction': 'LONG',
        'pair': 'UNKNOWN',
    }
    result = format_signal(signal)
    assert 'LONG' in result


@patch('telegram_bot.requests.post')
def test_send_telegram_message_success(mock_post):
    """Test successful Telegram message send."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.raise_for_status.return_value = None

    import telegram_bot
    telegram_bot.BOT_TOKEN = 'test:token'
    result = send_telegram_message('Test message')
    assert result is True


@patch('telegram_bot.requests.post')
def test_send_telegram_message_no_token(mock_post):
    """Test send fails gracefully when token is missing."""
    import telegram_bot
    telegram_bot.BOT_TOKEN = ''
    result = send_telegram_message('Test message')
    assert result is False


def test_send_signal():
    """Test sending a signal wraps properly."""
    signal = {
        'direction': 'LONG',
        'pair': 'TEST/USDT',
        'confidence': 95,
        'entry': 100,
        'take_profit': 120,
        'stop_loss': 90,
        'reasoning': 'Test signal.',
    }
    with patch('telegram_bot.send_telegram_message') as mock_send:
        mock_send.return_value = True
        result = send_signal(signal)
        assert result is True
        mock_send.assert_called_once()
        args = mock_send.call_args[0][0]
        assert 'TEST/USDT' in args