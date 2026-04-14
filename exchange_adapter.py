"""
DeepAlpha -- Exchange Adapter Layer

Provides a unified interface for trading on multiple exchanges.
Currently supported: Hyperliquid, Bitget, Binance Futures, Bybit.

Usage:
    from exchange_adapter import get_exchange

    exchange = get_exchange("bitget")
    exchange.connect()
    balance = exchange.get_balance()
"""

import os
import time
from abc import ABC, abstractmethod

import requests
from eth_account import Account
from hyperliquid.utils import constants
from hyperliquid.exchange import Exchange as HLExchange
from hyperliquid.info import Info as HLInfo


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class ExchangeAdapter(ABC):
    """
    Abstract interface that every exchange adapter must implement.

    All methods use a common data format so the trading bot does not need
    to know which exchange it is connected to.
    """

    @abstractmethod
    def connect(self) -> None:
        """Authenticate and establish the connection to the exchange."""
        ...

    @abstractmethod
    def get_balance(self) -> float:
        """Return total account equity in USD."""
        ...

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """
        Return open positions as a list of dicts:
        [
            {
                "coin": "BTC",
                "size": 0.1,          # positive = long, negative = short
                "entry": 65000.0,
                "side": "long",       # "long" or "short"
                "unrealized_pnl": 12.5,
            },
            ...
        ]
        """
        ...

    @abstractmethod
    def get_orderbook(self, coin: str) -> dict:
        """
        Return the top-of-book for *coin*.

        Returns
        -------
        {"bid": float, "ask": float, "mid": float}
        """
        ...

    @abstractmethod
    def get_candles(self, coin: str, interval: str = "1h",
                    limit: int = 200) -> list[dict]:
        """
        Return recent OHLCV candles.

        Each element:
        {"o": float, "h": float, "l": float, "c": float, "v": float}
        """
        ...

    @abstractmethod
    def place_limit_order(self, coin: str, side: str, size: float,
                          price: float) -> dict:
        """
        Place a limit order.

        Parameters
        ----------
        coin : str   -- e.g. "BTC"
        side : str   -- "buy" or "sell"
        size : float -- quantity in base asset
        price : float

        Returns
        -------
        {"success": bool, "order_id": str | None, "detail": str}
        """
        ...

    @abstractmethod
    def place_market_order(self, coin: str, side: str,
                           size: float) -> dict:
        """
        Place a market order.

        Returns
        -------
        {"success": bool, "order_id": str | None, "detail": str}
        """
        ...

    @abstractmethod
    def cancel_order(self, coin: str, oid: str) -> None:
        """Cancel an open order by its id."""
        ...

    @abstractmethod
    def close_position(self, coin: str) -> dict:
        """
        Fully close an open position for *coin*.

        Returns
        -------
        {"success": bool, "detail": str}
        """
        ...

    @abstractmethod
    def set_leverage(self, coin: str, leverage: int) -> None:
        """Set leverage for *coin* on the exchange."""
        ...

    @abstractmethod
    def get_funding_rate(self, coin: str) -> float:
        """Return the current funding rate for *coin* (as a decimal)."""
        ...


# ---------------------------------------------------------------------------
# Hyperliquid
# ---------------------------------------------------------------------------

HL_INFO_URL = "https://api.hyperliquid.xyz/info"


class HyperliquidAdapter(ExchangeAdapter):
    """Adapter for Hyperliquid L1 perpetual futures."""

    def __init__(self) -> None:
        self.private_key: str = os.getenv("PRIVATE_KEY", "")
        self.wallet: str = os.getenv("WALLET_ADDRESS", "")
        self.info: HLInfo | None = None
        self.exchange: HLExchange | None = None

    # -- connection ---------------------------------------------------------

    def connect(self) -> None:
        if not self.private_key:
            raise ValueError("PRIVATE_KEY env var not set (required for Hyperliquid)")
        if not self.wallet:
            raise ValueError("WALLET_ADDRESS env var not set (required for Hyperliquid)")
        account = Account.from_key(self.private_key)
        self.info = HLInfo(constants.MAINNET_API_URL, skip_ws=True)
        self.exchange = HLExchange(account, constants.MAINNET_API_URL)

    # -- account info -------------------------------------------------------

    def get_balance(self) -> float:
        try:
            state = self.info.user_state(self.wallet)
            return float(state["marginSummary"]["accountValue"])
        except Exception:
            return 0.0

    def get_positions(self) -> list[dict]:
        try:
            state = self.info.user_state(self.wallet)
            positions: list[dict] = []
            for pos in state.get("assetPositions", []):
                p = pos["position"]
                size = float(p["szi"])
                if size != 0:
                    positions.append({
                        "coin": p["coin"],
                        "size": size,
                        "entry": float(p["entryPx"]),
                        "side": "long" if size > 0 else "short",
                        "unrealized_pnl": float(p["unrealizedPnl"]),
                    })
            return positions
        except Exception:
            return []

    # -- market data --------------------------------------------------------

    def get_orderbook(self, coin: str) -> dict:
        book = self.info.l2_snapshot(coin)
        best_bid = float(book["levels"][0][0]["px"])
        best_ask = float(book["levels"][1][0]["px"])
        return {"bid": best_bid, "ask": best_ask, "mid": (best_bid + best_ask) / 2}

    def get_candles(self, coin: str, interval: str = "1h",
                    limit: int = 200) -> list[dict]:
        end_ms = int(time.time() * 1000)
        interval_seconds = _parse_interval(interval)
        start_ms = end_ms - (limit * interval_seconds * 1000)
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
            },
        }
        resp = requests.post(HL_INFO_URL, json=payload, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        return [
            {"o": float(c["o"]), "h": float(c["h"]),
             "l": float(c["l"]), "c": float(c["c"]), "v": float(c["v"])}
            for c in raw
        ]

    def get_funding_rate(self, coin: str) -> float:
        try:
            payload = {"type": "metaAndAssetCtxs"}
            resp = requests.post(HL_INFO_URL, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            meta = data[0]
            ctxs = data[1]
            for i, asset in enumerate(meta["universe"]):
                if asset["name"] == coin:
                    return float(ctxs[i]["funding"])
        except Exception:
            pass
        return 0.0

    # -- order execution ----------------------------------------------------

    def place_limit_order(self, coin: str, side: str, size: float,
                          price: float) -> dict:
        try:
            is_buy = side == "buy"
            result = self.exchange.order(
                coin, is_buy, size, price,
                {"limit": {"tif": "Gtc"}},
            )
            if result["status"] == "ok":
                return {"success": True, "order_id": None, "detail": str(result)}
            return {"success": False, "order_id": None, "detail": str(result)}
        except Exception as e:
            return {"success": False, "order_id": None, "detail": str(e)}

    def place_market_order(self, coin: str, side: str,
                           size: float) -> dict:
        try:
            is_buy = side == "buy"
            # Use IOC limit with slippage to simulate market order
            book = self.get_orderbook(coin)
            slippage = 0.002
            if is_buy:
                price = book["ask"] * (1 + slippage)
            else:
                price = book["bid"] * (1 - slippage)
            price = _round_price(price)
            result = self.exchange.order(
                coin, is_buy, size, price,
                {"limit": {"tif": "Ioc"}},
            )
            if result["status"] == "ok":
                return {"success": True, "order_id": None, "detail": str(result)}
            return {"success": False, "order_id": None, "detail": str(result)}
        except Exception as e:
            return {"success": False, "order_id": None, "detail": str(e)}

    def cancel_order(self, coin: str, oid: str) -> None:
        self.exchange.cancel(coin, int(oid))

    def close_position(self, coin: str) -> dict:
        try:
            positions = self.get_positions()
            target = None
            for p in positions:
                if p["coin"] == coin:
                    target = p
                    break
            if target is None:
                return {"success": False, "detail": f"No open position for {coin}"}

            side = "sell" if target["side"] == "long" else "buy"
            return self.place_market_order(coin, side, abs(target["size"]))
        except Exception as e:
            return {"success": False, "detail": str(e)}

    def set_leverage(self, coin: str, leverage: int) -> None:
        self.exchange.update_leverage(leverage, coin, is_cross=True)


# ---------------------------------------------------------------------------
# Binance Futures (USDT-M) via ccxt
# ---------------------------------------------------------------------------

class BinanceAdapter(ExchangeAdapter):
    """Adapter for Binance USDT-M Futures using ccxt."""

    def __init__(self) -> None:
        self.api_key: str = os.getenv("BINANCE_API_KEY", "")
        self.api_secret: str = os.getenv("BINANCE_API_SECRET", "")
        self.testnet: bool = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
        self.client = None

    def connect(self) -> None:
        try:
            import ccxt
        except ImportError:
            raise ImportError(
                "ccxt is required for Binance support. "
                "Install it with: pip install ccxt"
            )
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "BINANCE_API_KEY and BINANCE_API_SECRET env vars are required"
            )
        self.client = ccxt.binance({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        if self.testnet:
            self.client.set_sandbox_mode(True)
        self.client.load_markets()

    # -- helpers ------------------------------------------------------------

    def _symbol(self, coin: str) -> str:
        """Convert 'BTC' to 'BTC/USDT:USDT' (ccxt linear perp format)."""
        return f"{coin}/USDT:USDT"

    # -- account info -------------------------------------------------------

    def get_balance(self) -> float:
        try:
            bal = self.client.fetch_balance({"type": "future"})
            return float(bal["total"].get("USDT", 0))
        except Exception:
            return 0.0

    def get_positions(self) -> list[dict]:
        try:
            raw = self.client.fetch_positions()
            positions: list[dict] = []
            for p in raw:
                size = float(p["contracts"] or 0) * float(p["contractSize"] or 1)
                if size == 0:
                    continue
                side_str = p["side"]  # "long" or "short"
                signed_size = size if side_str == "long" else -size
                positions.append({
                    "coin": p["symbol"].split("/")[0],
                    "size": signed_size,
                    "entry": float(p["entryPrice"] or 0),
                    "side": side_str,
                    "unrealized_pnl": float(p["unrealizedPnl"] or 0),
                })
            return positions
        except Exception:
            return []

    # -- market data --------------------------------------------------------

    def get_orderbook(self, coin: str) -> dict:
        book = self.client.fetch_order_book(self._symbol(coin), limit=5)
        bid = book["bids"][0][0]
        ask = book["asks"][0][0]
        return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2}

    def get_candles(self, coin: str, interval: str = "1h",
                    limit: int = 200) -> list[dict]:
        ohlcv = self.client.fetch_ohlcv(self._symbol(coin), interval, limit=limit)
        return [
            {"o": c[1], "h": c[2], "l": c[3], "c": c[4], "v": c[5]}
            for c in ohlcv
        ]

    def get_funding_rate(self, coin: str) -> float:
        try:
            data = self.client.fetch_funding_rate(self._symbol(coin))
            return float(data.get("fundingRate", 0))
        except Exception:
            return 0.0

    # -- order execution ----------------------------------------------------

    def place_limit_order(self, coin: str, side: str, size: float,
                          price: float) -> dict:
        try:
            order = self.client.create_limit_order(
                self._symbol(coin), side, size, price
            )
            return {
                "success": True,
                "order_id": order.get("id"),
                "detail": str(order),
            }
        except Exception as e:
            return {"success": False, "order_id": None, "detail": str(e)}

    def place_market_order(self, coin: str, side: str,
                           size: float) -> dict:
        try:
            order = self.client.create_market_order(
                self._symbol(coin), side, size
            )
            return {
                "success": True,
                "order_id": order.get("id"),
                "detail": str(order),
            }
        except Exception as e:
            return {"success": False, "order_id": None, "detail": str(e)}

    def cancel_order(self, coin: str, oid: str) -> None:
        self.client.cancel_order(oid, self._symbol(coin))

    def close_position(self, coin: str) -> dict:
        try:
            positions = self.get_positions()
            target = None
            for p in positions:
                if p["coin"] == coin:
                    target = p
                    break
            if target is None:
                return {"success": False, "detail": f"No open position for {coin}"}
            side = "sell" if target["side"] == "long" else "buy"
            return self.place_market_order(coin, side, abs(target["size"]))
        except Exception as e:
            return {"success": False, "detail": str(e)}

    def set_leverage(self, coin: str, leverage: int) -> None:
        try:
            self.client.set_leverage(leverage, self._symbol(coin))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Bybit (USDT Perpetual) via ccxt
# ---------------------------------------------------------------------------

class BybitAdapter(ExchangeAdapter):
    """Adapter for Bybit USDT perpetual futures using ccxt."""

    def __init__(self) -> None:
        self.api_key: str = os.getenv("BYBIT_API_KEY", "")
        self.api_secret: str = os.getenv("BYBIT_API_SECRET", "")
        self.testnet: bool = os.getenv("BYBIT_TESTNET", "false").lower() == "true"
        self.client = None

    def connect(self) -> None:
        try:
            import ccxt
        except ImportError:
            raise ImportError(
                "ccxt is required for Bybit support. "
                "Install it with: pip install ccxt"
            )
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "BYBIT_API_KEY and BYBIT_API_SECRET env vars are required"
            )
        self.client = ccxt.bybit({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "linear"},
        })
        if self.testnet:
            self.client.set_sandbox_mode(True)
        self.client.load_markets()

    # -- helpers ------------------------------------------------------------

    def _symbol(self, coin: str) -> str:
        """Convert 'BTC' to 'BTC/USDT:USDT'."""
        return f"{coin}/USDT:USDT"

    # -- account info -------------------------------------------------------

    def get_balance(self) -> float:
        try:
            bal = self.client.fetch_balance({"type": "contract"})
            return float(bal["total"].get("USDT", 0))
        except Exception:
            return 0.0

    def get_positions(self) -> list[dict]:
        try:
            raw = self.client.fetch_positions()
            positions: list[dict] = []
            for p in raw:
                size = float(p["contracts"] or 0) * float(p["contractSize"] or 1)
                if size == 0:
                    continue
                side_str = p["side"]
                signed_size = size if side_str == "long" else -size
                positions.append({
                    "coin": p["symbol"].split("/")[0],
                    "size": signed_size,
                    "entry": float(p["entryPrice"] or 0),
                    "side": side_str,
                    "unrealized_pnl": float(p["unrealizedPnl"] or 0),
                })
            return positions
        except Exception:
            return []

    # -- market data --------------------------------------------------------

    def get_orderbook(self, coin: str) -> dict:
        book = self.client.fetch_order_book(self._symbol(coin), limit=5)
        bid = book["bids"][0][0]
        ask = book["asks"][0][0]
        return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2}

    def get_candles(self, coin: str, interval: str = "1h",
                    limit: int = 200) -> list[dict]:
        ohlcv = self.client.fetch_ohlcv(self._symbol(coin), interval, limit=limit)
        return [
            {"o": c[1], "h": c[2], "l": c[3], "c": c[4], "v": c[5]}
            for c in ohlcv
        ]

    def get_funding_rate(self, coin: str) -> float:
        try:
            data = self.client.fetch_funding_rate(self._symbol(coin))
            return float(data.get("fundingRate", 0))
        except Exception:
            return 0.0

    # -- order execution ----------------------------------------------------

    def place_limit_order(self, coin: str, side: str, size: float,
                          price: float) -> dict:
        try:
            order = self.client.create_limit_order(
                self._symbol(coin), side, size, price
            )
            return {
                "success": True,
                "order_id": order.get("id"),
                "detail": str(order),
            }
        except Exception as e:
            return {"success": False, "order_id": None, "detail": str(e)}

    def place_market_order(self, coin: str, side: str,
                           size: float) -> dict:
        try:
            order = self.client.create_market_order(
                self._symbol(coin), side, size
            )
            return {
                "success": True,
                "order_id": order.get("id"),
                "detail": str(order),
            }
        except Exception as e:
            return {"success": False, "order_id": None, "detail": str(e)}

    def cancel_order(self, coin: str, oid: str) -> None:
        self.client.cancel_order(oid, self._symbol(coin))

    def close_position(self, coin: str) -> dict:
        try:
            positions = self.get_positions()
            target = None
            for p in positions:
                if p["coin"] == coin:
                    target = p
                    break
            if target is None:
                return {"success": False, "detail": f"No open position for {coin}"}
            side = "sell" if target["side"] == "long" else "buy"
            return self.place_market_order(coin, side, abs(target["size"]))
        except Exception as e:
            return {"success": False, "detail": str(e)}

    def set_leverage(self, coin: str, leverage: int) -> None:
        try:
            self.client.set_leverage(leverage, self._symbol(coin))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Bitget (USDT-M Futures) via ccxt
# ---------------------------------------------------------------------------

class BitgetAdapter(ExchangeAdapter):
    """Adapter for Bitget USDT-M Futures using ccxt."""

    def __init__(self) -> None:
        self.api_key: str = os.getenv("BITGET_API_KEY", "")
        self.api_secret: str = os.getenv("BITGET_SECRET", "")
        self.passphrase: str = os.getenv("BITGET_PASSPHRASE", "")
        self.client = None

    def connect(self) -> None:
        try:
            import ccxt
        except ImportError:
            raise ImportError(
                "ccxt is required for Bitget support. "
                "Install it with: pip install ccxt"
            )
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "BITGET_API_KEY, BITGET_SECRET, and BITGET_PASSPHRASE "
                "env vars are required"
            )
        self.client = ccxt.bitget({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "password": self.passphrase,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        self.client.load_markets()

    # -- helpers ------------------------------------------------------------

    def _symbol(self, coin: str) -> str:
        return f"{coin}/USDT:USDT"

    # -- account info -------------------------------------------------------

    def get_balance(self) -> float:
        try:
            bal = self.client.fetch_balance({"type": "swap"})
            return float(bal["total"].get("USDT", 0))
        except Exception:
            return 0.0

    def get_positions(self) -> list[dict]:
        try:
            raw = self.client.fetch_positions()
            positions: list[dict] = []
            for p in raw:
                size = abs(float(p.get("contracts", 0) or 0))
                if size == 0:
                    continue
                side_str = p.get("side", "long")
                positions.append({
                    "coin": p["symbol"].split("/")[0],
                    "size": size if side_str == "long" else -size,
                    "entry": float(p.get("entryPrice", 0) or 0),
                    "side": side_str,
                    "unrealized_pnl": float(p.get("unrealizedPnl", 0) or 0),
                })
            return positions
        except Exception:
            return []

    # -- market data --------------------------------------------------------

    def get_orderbook(self, coin: str) -> dict:
        book = self.client.fetch_order_book(self._symbol(coin), limit=5)
        bid = book["bids"][0][0]
        ask = book["asks"][0][0]
        return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2}

    def get_candles(self, coin: str, interval: str = "1h",
                    limit: int = 200) -> list[dict]:
        ohlcv = self.client.fetch_ohlcv(self._symbol(coin), interval, limit=limit)
        return [
            {"o": c[1], "h": c[2], "l": c[3], "c": c[4], "v": c[5]}
            for c in ohlcv
        ]

    def get_funding_rate(self, coin: str) -> float:
        try:
            data = self.client.fetch_funding_rate(self._symbol(coin))
            return float(data.get("fundingRate", 0))
        except Exception:
            return 0.0

    # -- order execution ----------------------------------------------------

    def place_limit_order(self, coin: str, side: str, size: float,
                          price: float) -> dict:
        try:
            order = self.client.create_limit_order(
                self._symbol(coin), side, size, price
            )
            return {
                "success": True,
                "order_id": order.get("id"),
                "detail": str(order),
            }
        except Exception as e:
            return {"success": False, "order_id": None, "detail": str(e)}

    def place_market_order(self, coin: str, side: str,
                           size: float) -> dict:
        try:
            order = self.client.create_market_order(
                self._symbol(coin), side, size
            )
            return {
                "success": True,
                "order_id": order.get("id"),
                "detail": str(order),
            }
        except Exception as e:
            return {"success": False, "order_id": None, "detail": str(e)}

    def cancel_order(self, coin: str, oid: str) -> None:
        self.client.cancel_order(oid, self._symbol(coin))

    def close_position(self, coin: str) -> dict:
        try:
            positions = self.get_positions()
            target = None
            for p in positions:
                if p["coin"] == coin:
                    target = p
                    break
            if target is None:
                return {"success": False, "detail": f"No open position for {coin}"}
            side = "sell" if target["side"] == "long" else "buy"
            return self.place_market_order(coin, side, abs(target["size"]))
        except Exception as e:
            return {"success": False, "detail": str(e)}

    def set_leverage(self, coin: str, leverage: int) -> None:
        try:
            self.client.set_leverage(leverage, self._symbol(coin),
                                     params={"marginCoin": "USDT"})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _parse_interval(interval: str) -> int:
    """Convert interval string like '1h', '15m', '1d' to seconds."""
    units = {"m": 60, "h": 3600, "d": 86400}
    unit = interval[-1]
    value = int(interval[:-1])
    return value * units.get(unit, 3600)


def _round_price(price: float) -> float:
    """Round price to a reasonable number of decimals."""
    if price > 1000:
        return round(price, 1)
    elif price > 1:
        return round(price, 4)
    return round(price, 6)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ADAPTERS: dict[str, type[ExchangeAdapter]] = {
    "hyperliquid": HyperliquidAdapter,
    "bitget": BitgetAdapter,
    "binance": BinanceAdapter,
    "bybit": BybitAdapter,
}


def get_exchange(name: str) -> ExchangeAdapter:
    """
    Return an exchange adapter instance by name.

    Parameters
    ----------
    name : str
        One of "hyperliquid", "binance", "bybit" (case-insensitive).

    Returns
    -------
    ExchangeAdapter
        An unconnected adapter -- call .connect() before use.

    Raises
    ------
    ValueError
        If the exchange name is not recognised.
    """
    key = name.strip().lower()
    cls = _ADAPTERS.get(key)
    if cls is None:
        supported = ", ".join(sorted(_ADAPTERS.keys()))
        raise ValueError(
            f"Unknown exchange '{name}'. Supported: {supported}"
        )
    return cls()
