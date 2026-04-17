"""
DeepAlpha FreqAI Plugin
=======================

A FreqAI-compatible model for Freqtrade implementing:
- Triple Barrier Labeling
- SHAP-based feature selection
- Meta-labeling for trade filtering
- Purged walk-forward cross-validation

Usage:
    from deepalpha_freqai import DeepAlphaModel

Note: DeepAlphaModel requires freqtrade to be installed.
Utilities (PurgedWalkForwardCV, apply_triple_barrier, select_features_by_shap)
are available without freqtrade.
"""

__version__ = "1.1.2"
__author__ = "DeepAlpha Team"


# ── Anonymous usage telemetry (opt-out: DEEPALPHA_NO_TELEMETRY=1) ─────────
def _ping():
    """Send a single anonymous ping on first import. No PII collected."""
    import os
    if os.environ.get("DEEPALPHA_NO_TELEMETRY", "").strip() in ("1", "true", "yes"):
        return
    try:
        import threading, platform, sys, hashlib, uuid

        # Anonymous machine ID (hash of MAC address — not reversible)
        raw = str(uuid.getnode()).encode()
        machine_id = hashlib.sha256(raw).hexdigest()[:16]

        params = {
            "v": __version__,
            "py": f"{sys.version_info.major}.{sys.version_info.minor}",
            "os": platform.system().lower(),
            "mid": machine_id,
        }

        def _do():
            try:
                import urllib.request, urllib.parse
                qs = urllib.parse.urlencode(params)
                url = f"https://deepalphabot.com/api/ping?{qs}"
                req = urllib.request.Request(url, method="GET")
                urllib.request.urlopen(req, timeout=3)
            except Exception:
                pass

        # Fire-and-forget in background thread (never blocks import)
        t = threading.Thread(target=_do, daemon=True)
        t.start()
    except Exception:
        pass

_ping()


def __getattr__(name):
    """Lazy import to avoid crashing when freqtrade isn't installed."""
    if name == "DeepAlphaModel":
        from .deepalpha_model import DeepAlphaModel
        return DeepAlphaModel
    if name in ("PurgedWalkForwardCV", "apply_triple_barrier", "select_features_by_shap"):
        from .deepalpha_model import (
            PurgedWalkForwardCV,
            apply_triple_barrier,
            select_features_by_shap,
        )
        return {
            "PurgedWalkForwardCV": PurgedWalkForwardCV,
            "apply_triple_barrier": apply_triple_barrier,
            "select_features_by_shap": select_features_by_shap,
        }[name]
    raise AttributeError(f"module 'deepalpha_freqai' has no attribute {name!r}")


__all__ = [
    "DeepAlphaModel",
    "PurgedWalkForwardCV",
    "apply_triple_barrier",
    "select_features_by_shap",
]
