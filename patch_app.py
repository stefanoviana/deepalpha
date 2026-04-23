#!/usr/bin/env python3
"""Patch script for /root/deepalpha-cloud/app.py - 4 fixes (P4, S2, S3, S7)."""

FILE = "/root/deepalpha-cloud/app.py"

with open(FILE, "r") as f:
    code = f.read()

original = code
changes = 0

# ============================================================================
# P4: Backtest blocks event loop - wrap in run_in_executor
# ============================================================================
# The backtest endpoint does heavy numpy computation synchronously.
# We need to extract the computation into a sync function and wrap it.

# Find the backtest endpoint and wrap the heavy part
old_bt_start = '''@app.post("/api/backtest")
async def run_backtest(
    body: BacktestReq,
    user: User = Depends(get_current_user),
):
    days = min(body.days, BACKTEST_MAX_DAYS)

    initial_capital = body.initial_capital
    margin_per_trade = body.margin_per_trade

    # Check cache
    cache_key = f"portfolio_{days}_{int(initial_capital)}_{int(margin_per_trade)}_{body.leverage}_{body.max_positions}_{int(body.confidence_threshold)}"
    now = time.time()
    if cache_key in _backtest_cache:
        cached_time, cached_result = _backtest_cache[cache_key]
        if now - cached_time < BACKTEST_CACHE_TTL:
            logger.info("Portfolio backtest cache hit: %s", cache_key)
            return cached_result

    # Load model
    model_data = _get_bt_model("1h")
    if not model_data:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Model not loaded")

    fear_greed = _bt_get_fear_greed()

    # Fetch candles for all portfolio coins
    limit = days * 24 + 200
    end_time = int(now * 1000)
    start_time = end_time - ((days * 24 + 200) * 3600 * 1000)

    coin_data = {}
    import asyncio as _aio

    async with httpx.AsyncClient(timeout=60) as client:
        for coin in PORTFOLIO_BT_COINS:
            coin_name, records = await _fetch_coin_candles(client, coin, start_time, end_time, limit)
            if len(records) >= 50:
                coin_data[coin_name] = {
                    "timestamps": [r["timestamp"] for r in records],
                    "closes": np.array([r["close"] for r in records], dtype=float),
                    "opens": np.array([r["open"] for r in records], dtype=float),
                    "highs": np.array([r["high"] for r in records], dtype=float),
                    "lows": np.array([r["low"] for r in records], dtype=float),
                    "volumes": np.array([r["volume"] for r in records], dtype=float),
                }
            await _aio.sleep(0.15)

    if not coin_data:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to fetch candle data")'''

new_bt_start = '''def _run_backtest_sync(coin_data, model_data, fear_greed, body, initial_capital, margin_per_trade, days):
    """Sync backtest computation - runs in executor to avoid blocking event loop (P4)."""'''

# Instead of the complex extraction, let's take a simpler approach:
# Add a helper that wraps the heavy computation part

# Actually, the simplest approach is to wrap the ENTIRE computation section
# (after data fetching) in run_in_executor. Let me find the computation boundary.

# The data fetching (async HTTP) ends at "if not coin_data:" check.
# Everything after that is pure numpy computation.
# We'll extract lines from "btc_closes = ..." to "return result" into _run_backtest_sync

old_compute_start = "    btc_closes = coin_data.get(\"BTC\", {}).get(\"closes\", np.array([0.0]))"

old_compute_and_return = """    btc_closes = coin_data.get("BTC", {}).get("closes", np.array([0.0]))

    # Unified timeline from BTC
    ref_coin = "BTC" if "BTC" in coin_data else list(coin_data.keys())[0]"""

# Simpler approach: just wrap the computation portion after data fetch
# Find the computation block boundary

if old_compute_and_return in code:
    # Find everything from btc_closes to the end of the function (return result)
    # We'll add run_in_executor around it

    # Replace the computation start to use run_in_executor
    old_section = """    if not coin_data:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to fetch candle data")

    btc_closes = coin_data.get("BTC", {}).get("closes", np.array([0.0]))"""

    new_section = """    if not coin_data:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to fetch candle data")

    # P4: Run heavy computation in executor to avoid blocking event loop
    import functools
    _bt_func = functools.partial(
        _run_backtest_compute, coin_data, model_data, fear_greed, body, initial_capital, margin_per_trade, days, now
    )
    result = await asyncio.get_event_loop().run_in_executor(None, _bt_func)

    _backtest_cache[cache_key] = (now, result)
    logger.info("Portfolio backtest done: %dd => %d trades, %.1f%% WR, $%.2f PnL, %d coins",
                days, result["total_trades"], result["win_rate"], result["total_pnl"], len(coin_data))

    return result


def _run_backtest_compute(coin_data, model_data, fear_greed, body, initial_capital, margin_per_trade, days, now):
    \"\"\"Heavy backtest computation extracted to run in thread executor (P4).\"\"\"
    btc_closes = coin_data.get("BTC", {}).get("closes", np.array([0.0]))"""

    if old_section in code:
        code = code.replace(old_section, new_section, 1)
        changes += 1
        print("P4a: run_in_executor wrapper added")

    # Now remove the old cache store and return at the end since we moved it
    old_tail = """    _backtest_cache[cache_key] = (now, result)
    logger.info("Portfolio backtest done: %dd => %d trades, %.1f%% WR, $%.2f PnL, %d coins",
                days, total_trades, win_rate, cumulative_pnl_dollars, len(coin_data))

    return result"""

    new_tail = """    return result"""

    if old_tail in code:
        code = code.replace(old_tail, new_tail, 1)
        changes += 1
        print("P4b: Moved cache store to caller")
else:
    print("P4: SKIPPED - compute start not found")

# ============================================================================
# S2: Admin access too broad - ONLY user.id == 4
# ============================================================================
old_admin = '''def _require_admin(user: User):
    """Raise 403 if user is not admin (Stefano: id=4 or plan=lifetime)."""
    if user.id != 4 and user.plan != PlanType.lifetime:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access only")'''

new_admin = '''def _require_admin(user: User):
    """Raise 403 if user is not admin (Stefano: id=4 only). S2 security fix."""
    if user.id != 4:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access only")'''

if old_admin in code:
    code = code.replace(old_admin, new_admin, 1)
    changes += 1
    print("S2: Admin restricted to user.id == 4 only")

# ============================================================================
# S3: Gumroad webhook - add secret verification
# ============================================================================
old_gumroad_seller = 'GUMROAD_SELLER_ID = os.environ.get("GUMROAD_SELLER_ID", "xq0fxoZqWGjHa0VnxMs72w==")'

new_gumroad_seller = '''GUMROAD_SELLER_ID = os.environ.get("GUMROAD_SELLER_ID", "xq0fxoZqWGjHa0VnxMs72w==")
GUMROAD_WEBHOOK_SECRET = os.environ.get("GUMROAD_WEBHOOK_SECRET", "")  # S3: shared secret for webhook verification'''

if old_gumroad_seller in code:
    code = code.replace(old_gumroad_seller, new_gumroad_seller, 1)
    changes += 1
    print("S3a: GUMROAD_WEBHOOK_SECRET env var added")

# Add secret check in the webhook endpoint
old_webhook = '''@app.post("/api/webhook/gumroad")
async def gumroad_webhook(body: GumroadWebhook, db: AsyncSession = Depends(get_db)):
    import secrets as _secrets

    # verify seller_id to prevent spoofed webhooks
    if body.seller_id != GUMROAD_SELLER_ID:
        logger.warning("Gumroad webhook rejected \xe2\x80\x94 bad seller_id: %s", body.seller_id)
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid seller")'''

new_webhook = '''@app.post("/api/webhook/gumroad")
async def gumroad_webhook(body: GumroadWebhook, request: Request, db: AsyncSession = Depends(get_db)):
    import secrets as _secrets

    # S3: Verify webhook secret if configured
    if GUMROAD_WEBHOOK_SECRET:
        _wh_secret = request.headers.get("X-Gumroad-Secret", "")
        if not _secrets.compare_digest(_wh_secret, GUMROAD_WEBHOOK_SECRET):
            logger.warning("Gumroad webhook rejected - bad secret from IP %s", get_client_ip(request))
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid webhook secret")

    # verify seller_id to prevent spoofed webhooks
    if body.seller_id != GUMROAD_SELLER_ID:
        logger.warning("Gumroad webhook rejected \xe2\x80\x94 bad seller_id: %s", body.seller_id)
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid seller")'''

if old_webhook in code:
    code = code.replace(old_webhook, new_webhook, 1)
    changes += 1
    print("S3b: Webhook secret verification added")
else:
    # Try with em dash as unicode
    old_webhook2 = old_webhook.replace("\xe2\x80\x94", "\u2014")
    if old_webhook2 in code:
        new_webhook2 = new_webhook.replace("\xe2\x80\x94", "\u2014")
        code = code.replace(old_webhook2, new_webhook2, 1)
        changes += 1
        print("S3b: Webhook secret verification added (unicode variant)")
    else:
        print("S3b: SKIPPED - webhook block not found")

# ============================================================================
# S7: Download endpoint crash - user.get() -> user.plan
# ============================================================================
old_download = '''    plan = user.get("plan", "free_trial")
    if plan not in ("pro", "lifetime"):'''

new_download = '''    plan = getattr(user, "plan", "free_trial")
    # Convert enum to string for comparison
    plan_str = plan.value if hasattr(plan, "value") else str(plan)
    if plan_str not in ("pro", "lifetime"):'''

if old_download in code:
    code = code.replace(old_download, new_download, 1)
    changes += 1
    print("S7: download_bot user.get() -> getattr(user, 'plan') fixed")

# ============================================================================
# Final write
# ============================================================================
if code != original:
    with open(FILE, "w") as f:
        f.write(code)
    print(f"\nSUCCESS: {changes} changes applied to app.py ({len(original)} -> {len(code)} bytes)")
else:
    print("\nWARNING: No changes were made!")
