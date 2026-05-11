"""
DeepAlpha Telegram Signals Bot

Broadcasts free AI-generated trading signals to a public Telegram channel.
Reads signals from the DeepAlpha engine and posts them formatted with:
- LONG/SHORT direction
- Coin/asset pair
- Confidence score
- Entry price
- Take profit / Stop loss levels
"""

import os
import json
import logging
import requests
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger('deepalpha.telegram')

# Telegram configuration
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID', '@DeepAlphaSignals')

# Signal template
SIGNAL_TEMPLATE = """🚀 *DeepAlpha Signal*

*{direction}* {pair}
Confidence: {confidence}%
Entry: ${entry}
Take Profit: ${tp}
Stop Loss: ${sl}

{reasoning}

#DeepAlpha #CryptoSignals #AI
"""


def send_telegram_message(text: str, parse_mode: str = 'Markdown') -> bool:
    """Send a message to the configured Telegram channel."""
    if not BOT_TOKEN:
        logger.error('TELEGRAM_BOT_TOKEN not configured')
        return False

    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    payload = {
        'chat_id': CHANNEL_ID,
        'text': text,
        'parse_mode': parse_mode,
        'disable_web_page_preview': True,
    }

    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        logger.info('Signal sent to %s', CHANNEL_ID)
        return True
    except requests.RequestException as e:
        logger.error('Failed to send Telegram message: %s', e)
        return False


def format_signal(signal: dict) -> str:
    """Format a trading signal dict into a Telegram message."""
    return SIGNAL_TEMPLATE.format(
        direction='🟢 LONG' if signal.get('direction', '').upper() == 'LONG' else '🔴 SHORT',
        pair=signal.get('pair', 'UNKNOWN'),
        confidence=signal.get('confidence', 0),
        entry=signal.get('entry', 0),
        tp=signal.get('take_profit', 0),
        sl=signal.get('stop_loss', 0),
        reasoning=signal.get('reasoning', 'AI analysis complete.')
    )


def send_signal(signal: dict) -> bool:
    """Send a structured trading signal to the Telegram channel."""
    message = format_signal(signal)
    return send_telegram_message(message)


def send_status(message: str) -> bool:
    """Send a status/info message to the channel."""
    return send_telegram_message(f'📊 *DeepAlpha Status*\n{message}')


def send_error(message: str) -> bool:
    """Send an error alert to the channel."""
    return send_telegram_message(f'⚠️ *DeepAlpha Alert*\n{message}')


if __name__ == '__main__':
    # Test signal
    test_signal = {
        'direction': 'LONG',
        'pair': 'BTC/USDT',
        'confidence': 87,
        'entry': 84500,
        'take_profit': 92000,
        'stop_loss': 80000,
        'reasoning': 'Strong bullish divergence detected on 4H RSI with increasing volume. Key resistance broken.'
    }
    success = send_signal(test_signal)
    print(f'Test signal sent: {success}')