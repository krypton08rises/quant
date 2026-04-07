Category 1: RAW MARKET DATA (Non-Stationary)

Action: DROP from Model Input (Use only for calculation/metadata)

    open, high, low, close: Absolute levels drift over time.

    shifted_close: Just a lagged price.

    ema_12, ema_26: Moving averages scale with price.

    bb_mid, bb_upper, bb_lower: Bollinger bands scale with price.

    atr: Volatility in rupee terms (e.g., ₹15) scales with price.

    upper_shadow, lower_shadow, tick_body: Rupee values.

Category 2: MOMENTUM & OSCILLATORS (Stationary - KEEP)

Action: Direct Input to Model

    rsi: Bounded 0-100. Perfect.

    bb_pct: Bounded 0-1 (mostly). Represents position relative to bands. Perfect.

    percent_change: This is your "Stationary Price." Keep.

    macd_hist: Usually stationary enough (divergence), though normalizing it by price is safer.
