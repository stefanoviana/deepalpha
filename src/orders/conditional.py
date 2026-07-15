"""Conditional / stop-loss order helpers (OKX/Binance scaffolding)."""
from dataclasses import dataclass
@dataclass
class ConditionalOrder:
    symbol: str
    side: str
    stop_price: float
    quantity: float
    order_type: str = "STOP_MARKET"
def to_binance_payload(o: ConditionalOrder) -> dict:
    return {
        "symbol": o.symbol,
        "side": o.side.upper(),
        "type": o.order_type,
        "stopPrice": str(o.stop_price),
        "quantity": str(o.quantity),
    }
def to_okx_payload(o: ConditionalOrder) -> dict:
    return {
        "instId": o.symbol,
        "tdMode": "cash",
        "side": o.side.lower(),
        "ordType": "conditional",
        "slTriggerPx": str(o.stop_price),
        "sz": str(o.quantity),
    }
