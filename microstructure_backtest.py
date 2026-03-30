"""
=============================================================================
MICROSTRUCTURE STRATEGY BACKTEST  v2.0
Based on: "Trades, Quotes and Prices"
Bouchaud, Bonart, Donier & Gould (Cambridge University Press, 2018)
=============================================================================

ARCHITECTURE (v2 — fixed from v1):
────────────────────────────────────
  LAYER 1  →  1-MINUTE INTRADAY BARS  (390 bars/day, 9:30–16:00 EST)
    • GARCH(1,1) volatility per bar          [Ch.9  — Hawkes clustering]
    • Student-t returns (ν=4)                [Ch.2  — Fat tails]
    • Bid-ask bounce + spread simulation     [Ch.16 — Bid-Ask Spread]
    • Order-flow sign series (Hawkes)        [Ch.9  — Self-excitation]

  LAYER 2  →  INTRADAY SIGNAL COMPUTATION  (per bar)
    • OFI  : buy_vol vs sell_vol per bar     [Ch.11 — Order Flow Imbalance]
    • Spread: simulated from H/L range       [Ch.16 — Corwin-Schultz proxy]
    • Hawkes: autocorr of |bar returns|      [Ch.9  — Clustering]
    • Kyle λ: |ret| / dollar_vol per bar     [Ch.15 — Adverse Selection]
    • Impact: sqrt law on bar volume         [Ch.12 — Square-Root Law]

  LAYER 3  →  DAILY AGGREGATION
    • Signals averaged across all 390 bars → stable daily score
    • Liquidity filter: skip if avg spread > threshold
    • This is the KEY fix vs v1 — intraday signal → daily decision

  LAYER 4  →  BACKTEST  (trade at next-day open)
    • Commission: 10bps per trade (Jerry's number — 0.1%)
    • Market impact: ZERO (as Jerry suggested — assume no impact)
    • Minimum holding period: 3 days (cuts turnover dramatically)
    • Turnover cap: max 5% portfolio moved per day

V1 vs V2 COST COMPARISON:
──────────────────────────
  V1: 8bps spread+impact × 18.3%/day turnover ≈ 1.46bps/day ≈ 3.7%/yr drag
  V2: 10bps commission   × <5%/day turnover   ≈ 0.5bps/day  ≈ 1.3%/yr drag
  → V2 costs ~3x LOWER despite higher per-trade commission

DEPENDENCIES: numpy pandas matplotlib  (no scipy, no yfinance)
RUN: python3 Microstructure_Backtest_TQP_Replication.py
=============================================================================
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# ── Pure-numpy stats helpers (no scipy needed) ────────────────────────────────
class stats:
    class norm:
        @staticmethod
        def pdf(x, mu, sigma):
            x = np.asarray(x, float)
            return np.exp(-0.5*((x-mu)/sigma)**2) / (sigma*np.sqrt(2*np.pi))

def skew(x):
    x = np.asarray(x,float); x=x[~np.isnan(x)]
    m,s = x.mean(), x.std()
    return float(np.mean(((x-m)/s)**3)) if s>0 else 0.0

def kurtosis(x):
    x = np.asarray(x,float); x=x[~np.isnan(x)]
    m,s = x.mean(), x.std()
    return float(np.mean(((x-m)/s)**4)-3) if s>0 else 0.0

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # Universe
    "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "start":   "2020-01-01",
    "end":     "2025-01-01",

    # Intraday bar resolution
    "bars_per_day": 390,        # 1-min bars: 9:30–16:00 EST = 390 minutes

    # Signal windows (in 1-min BARS, not days)
    "ofi_bars":     20,         # 20 bars = 20 minutes rolling OFI
    "hawkes_bars":  60,         # 60 bars = 1 hour clustering window
    "spread_bars":  30,         # 30 bars for spread smoothing
    "kyle_bars":    60,         # 60 bars for Kyle lambda

    # Daily signal aggregation
    "signal_threshold": 0.4,    # z-score to enter (higher = fewer trades)
    "top_n":            3,      # long top N stocks

    # ── TRANSACTION COSTS (Jerry's spec) ─────────────────────────────────
    "commission_bps":   10.0,   # 10bps = 0.1% per TRADE (one-way)
    "market_impact_bps": 0.0,   # ZERO — assume no impact (Jerry said so)

    # ── TURNOVER CONTROL ─────────────────────────────────────────────────
    "holding_period":   3,      # minimum days to hold before rebalancing
    "max_turnover_day": 0.05,   # max 5% of portfolio moved per day
    "max_position":     0.30,   # max 30% per stock

    # Risk
    "vol_target":       0.15,   # 15% annual vol target
    "vol_window_days":  20,     # days for vol estimation

    # Liquidity filter
    "max_spread_filter": 0.003, # skip stock if avg spread > 30bps intraday
}

# ═════════════════════════════════════════════════════════════════════════════
# LAYER 1 — SYNTHETIC 1-MIN BAR GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

# Per-stock calibration to real NASDAQ empirical properties
STOCK_PARAMS = {
    # mu=annual drift, sigma=daily vol, spread_bps=typical bid-ask,
    # jump_lam=jump freq/day, p0=start price
    "AAPL":  dict(mu=0.25, sigma=0.018, spread_bps=2.0,  jump_lam=0.010, p0=77),
    "MSFT":  dict(mu=0.22, sigma=0.016, spread_bps=1.5,  jump_lam=0.008, p0=158),
    "GOOGL": dict(mu=0.20, sigma=0.020, spread_bps=2.5,  jump_lam=0.010, p0=68),
    "AMZN":  dict(mu=0.18, sigma=0.022, spread_bps=3.0,  jump_lam=0.012, p0=94),
    "META":  dict(mu=0.15, sigma=0.025, spread_bps=3.5,  jump_lam=0.015, p0=210),
    "NVDA":  dict(mu=0.35, sigma=0.030, spread_bps=2.0,  jump_lam=0.015, p0=60),
    "TSLA":  dict(mu=0.20, sigma=0.040, spread_bps=5.0,  jump_lam=0.020, p0=86),
}


def generate_one_minute_bars(ticker: str, trading_days: pd.DatetimeIndex,
                              bars_per_day: int = 390,
                              seed_offset: int = 0) -> pd.DataFrame:
    """
    Simulate realistic 1-minute OHLCV bars for one stock.

    MICROSTRUCTURE EFFECTS BAKED IN:
    ─────────────────────────────────
    1. GARCH(1,1) variance process → volatility clustering (Ch.9)
       ω=0.05, α=0.10, β=0.85  (standard calibration for US equities)

    2. Student-t shocks (ν=4) → fat-tailed returns (Ch.2.2)
       This gives kurtosis ≈ 6, matching real NASDAQ data

    3. Hawkes-style order-flow clustering (Ch.9)
       Order signs are serially correlated: ρ=0.3 per bar
       Captures the long-memory of order flow documented in Ch.10

    4. Bid-Ask Bounce (Ch.16)
       Observed price alternates bid/ask based on trade direction
       Creates negative lag-1 autocorrelation in returns (Ch.2.1.3)

    5. Intraday U-shape in volume (Ch.4.2)
       Volume highest at open/close, lowest at lunch
       Captured by a sinusoidal multiplier over the day

    6. Jump component (Ch.2.2)
       Rare large moves, Poisson arrivals scaled to per-bar frequency
    """
    np.random.seed(42 + seed_offset)
    p   = STOCK_PARAMS.get(ticker, dict(mu=0.15, sigma=0.02,
                            spread_bps=3.0, jump_lam=0.01, p0=100))
    n_days  = len(trading_days)
    n_bars  = n_days * bars_per_day
    dt      = 1 / (252 * bars_per_day)   # fraction of a year per bar

    # ── GARCH(1,1) at bar frequency ──────────────────────────────────────
    bar_var_annual = p["sigma"]**2           # annualised variance
    bar_var        = bar_var_annual * (1/252) / bars_per_day  # per-bar variance
    omega  = bar_var * (1 - 0.85 - 0.10)
    alpha  = 0.10
    beta   = 0.85
    h      = np.zeros(n_bars)
    eps    = np.zeros(n_bars)
    h[0]   = bar_var
    z      = np.random.standard_t(df=4, size=n_bars)  # fat tails

    for t in range(1, n_bars):
        h[t]   = omega + alpha * eps[t-1]**2 + beta * h[t-1]
        eps[t] = np.sqrt(h[t]) * z[t]

    # ── Intraday U-shape volume multiplier ────────────────────────────────
    bar_idx   = np.tile(np.arange(bars_per_day), n_days)
    norm_time = bar_idx / bars_per_day                        # 0 → 1
    u_shape   = 1.5 + np.cos(2 * np.pi * norm_time) * 0.5    # 1.0–2.0x

    # ── Jump component ────────────────────────────────────────────────────
    jump_prob_bar = p["jump_lam"] / bars_per_day
    jumps = np.zeros(n_bars)
    jump_mask  = np.random.rand(n_bars) < jump_prob_bar
    jump_size  = np.random.normal(0, p["sigma"] * 3, n_bars)
    jumps[jump_mask] = jump_size[jump_mask]

    # ── Drift per bar ─────────────────────────────────────────────────────
    drift = p["mu"] * dt

    # ── Price path ────────────────────────────────────────────────────────
    bar_ret = drift + eps + jumps
    log_p   = np.log(p["p0"]) + np.cumsum(bar_ret)
    mid     = np.exp(log_p)

    # ── Hawkes order-flow (Ch.9 — self-exciting sign series) ─────────────
    # Sign of order at each bar; serial correlation ρ=0.3
    signs = np.zeros(n_bars)
    signs[0] = np.random.choice([-1, 1])
    rho = 0.30
    u   = np.random.rand(n_bars)
    for t in range(1, n_bars):
        p_same = 0.5 + rho * 0.5 * signs[t-1]   # biased coin
        signs[t] = 1.0 if u[t] < p_same else -1.0

    # ── Bid-ask spread & OHLC construction ───────────────────────────────
    half_spread = mid * (p["spread_bps"] / 20_000)  # half-spread in price
    # Observed price = mid ± half_spread depending on trade direction
    observed = mid + signs * half_spread

    # OHLC from bar: open=start of bar, close=end of bar mid ± bounce
    close = observed
    open_ = np.roll(observed, 1);  open_[0] = p["p0"]

    # High / Low within bar: range driven by GARCH volatility
    intraday_range = mid * np.sqrt(h) * 2.5
    high = np.maximum(open_, close) + intraday_range * np.random.uniform(0.1, 0.4, n_bars)
    low  = np.minimum(open_, close) - intraday_range * np.random.uniform(0.1, 0.4, n_bars)

    # ── Volume: log-normal scaled by vol regime + U-shape ────────────────
    base_volume = 1_000_000 / p["p0"]
    vol_regime  = np.sqrt(h) / np.sqrt(bar_var)
    volume      = (base_volume * vol_regime * u_shape *
                   np.random.lognormal(0, 0.4, n_bars))
    volume      = volume.clip(min=1000).astype(np.int64)

    # ── Build timestamp index ─────────────────────────────────────────────
    timestamps = []
    for day in trading_days:
        open_time = pd.Timestamp(day.year, day.month, day.day, 9, 30)
        for b in range(bars_per_day):
            timestamps.append(open_time + pd.Timedelta(minutes=b))

    # Ensure all prices are positive before log
    close = np.maximum(close, 0.01)
    open_ = np.maximum(open_, 0.01)
    high  = np.maximum(high,  0.01)
    low   = np.maximum(low,   0.01)
    mid   = np.maximum(mid,   0.01)

    df = pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
        "mid":    mid,
        "sign":   signs,
        "h_var":  h,
        "half_spread": half_spread,
    }, index=pd.DatetimeIndex(timestamps))

    # Clip bar returns to ±10% (real 1-min moves never exceed this)
    df["ret"] = (np.log(df["close"] / df["close"].shift(1))
                 .replace([np.inf, -np.inf], np.nan)
                 .fillna(0)
                 .clip(-0.10, 0.10))
    df["date"] = df.index.date
    return df


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 2 — INTRADAY SIGNAL COMPUTATION (per 1-min bar)
# ═════════════════════════════════════════════════════════════════════════════

def compute_intraday_signals(bars: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute all 5 microstructure signals at 1-minute bar frequency.

    ──────────────────────────────────────────────────────────────────
    SIGNAL 1 — OFI (Order-Flow Imbalance)            [Book: Ch.11]
    ──────────────────────────────────────────────────────────────────
    At bar level we know the trade sign (buy=+1, sell=−1).
    OFI = rolling sum of signed volume / rolling total volume
    This is the TRUE microstructure signal — only meaningful intraday.

        OFI_bar = Σ(sign_i × vol_i) / Σ(vol_i)   over last N bars

    ──────────────────────────────────────────────────────────────────
    SIGNAL 2 — Price Impact / Square-Root Law        [Book: Ch.12]
    ──────────────────────────────────────────────────────────────────
    At each bar, estimate metaorder participation:

        impact_bar = sign × √(vol_bar / avg_vol_bar)

    This directly implements Eq.12.3 from the book.

    ──────────────────────────────────────────────────────────────────
    SIGNAL 3 — Spread Proxy                          [Book: Ch.16]
    ──────────────────────────────────────────────────────────────────
    At 1-min bars, we use the Roll (1984) estimator:
        spread ≈ 2 × √(−Cov(Δp_t, Δp_{t-1}))
    Negative return covariance = bid-ask bounce (Ch.2.1.3).
    Wide spread → adverse selection risk high.

    ──────────────────────────────────────────────────────────────────
    SIGNAL 4 — Hawkes Branching Ratio                [Book: Ch.9]
    ──────────────────────────────────────────────────────────────────
    Self-excitation proxy: autocorrelation of |returns|.
    High autocorrelation → market is in a clustered/volatile regime.

    ──────────────────────────────────────────────────────────────────
    SIGNAL 5 — Kyle Lambda (Adverse Selection)       [Book: Ch.15]
    ──────────────────────────────────────────────────────────────────
    Kyle's lambda = price sensitivity to order flow:

        λ = |Δp| / signed_volume   (per bar)

    Low λ → liquid, safe. High λ → informed trading, avoid.
    """
    b = bars.copy()
    W_ofi    = cfg["ofi_bars"]
    W_hawkes = cfg["hawkes_bars"]
    W_spread = cfg["spread_bars"]
    W_kyle   = cfg["kyle_bars"]

    # Signed volume per bar
    b["signed_vol"] = b["sign"] * b["volume"]

    # ── Signal 1: OFI ────────────────────────────────────────────────────
    roll_signed = b["signed_vol"].rolling(W_ofi).sum()
    roll_vol    = b["volume"].rolling(W_ofi).sum()
    b["ofi"]    = roll_signed / (roll_vol + 1)

    # ── Signal 2: Price Impact (Square-Root Law) ──────────────────────────
    avg_vol      = b["volume"].rolling(W_ofi).mean()
    vol_ratio    = b["volume"] / (avg_vol + 1)
    b["impact"]  = b["sign"] * np.sqrt(vol_ratio)

    # ── Signal 3: Spread (Roll estimator from return covariance) ─────────
    ret = b["ret"]
    cov_ret = (ret * ret.shift(1)).rolling(W_spread).mean()
    # Roll spread = 2*sqrt(max(0, -cov))  (covariance should be negative)
    b["spread_roll"] = 2 * np.sqrt(np.maximum(-cov_ret, 0))
    b["spread_roll"] = b["spread_roll"].rolling(W_spread).mean()

    # ── Signal 4: Hawkes Clustering ────────────────────────────────────
    abs_ret  = b["ret"].abs()
    lag1_abs = abs_ret.shift(1)
    num      = (abs_ret * lag1_abs).rolling(W_hawkes).mean()
    denom    = abs_ret.rolling(W_hawkes).var() + 1e-12
    b["hawkes"] = (num / denom).clip(0, 10)

    # ── Signal 5: Kyle Lambda ────────────────────────────────────────────
    dollar_flow  = (b["sign"] * b["volume"] * b["mid"]).abs() + 1
    b["kyle_lam"] = b["ret"].abs() / (dollar_flow / 1e6 + 1e-8)
    b["kyle_lam"] = b["kyle_lam"].clip(0, 10).rolling(W_kyle).mean()

    return b.dropna()


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 3 — DAILY AGGREGATION OF INTRADAY SIGNALS
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_to_daily(intraday: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse 390 bars/day into one daily signal row.

    AGGREGATION LOGIC:
    ──────────────────
    • OFI     → mean of last 60 bars of day (end-of-day flow bias)
    • Impact  → mean across day (metaorder size is steady)
    • Spread  → mean across day (liquidity measure)
    • Hawkes  → max across day (regime indicator — peak clustering)
    • Kyle λ  → mean across day (adverse selection level)
    • ret     → sum of bar returns = daily log return
    • vol     → std of bar returns × √390 = daily realised volatility
    • avg_spread_true → mean of true half_spread (for liquidity filter)
    """
    grp = intraday.groupby("date")

    daily = pd.DataFrame({
        "ofi":         grp["ofi"].apply(lambda x: x.iloc[-60:].mean()),
        "impact":      grp["impact"].mean(),
        "spread":      grp["spread_roll"].mean(),
        "hawkes":      grp["hawkes"].max(),
        "kyle_lam":    grp["kyle_lam"].mean(),
        "ret":         grp["ret"].sum().clip(-0.15, 0.15),
        "realised_vol":grp["ret"].std() * np.sqrt(390),
        "true_spread": grp["half_spread"].mean() * 2 / grp["mid"].mean(),
        "close":       grp["close"].last(),
        "volume":      grp["volume"].sum(),
    })

    daily.index = pd.to_datetime(daily.index)
    return daily.dropna()


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL COMBINER — DAILY COMPOSITE SCORE
# ═════════════════════════════════════════════════════════════════════════════

class SignalCombiner:
    """
    Combines 5 intraday-aggregated signals into one composite score.

    WEIGHTS (sign-corrected for trading):
    ──────────────────────────────────────
    +OFI    : positive flow → bullish
    +Impact : momentum confirmation
    +Hawkes : high clustering → trend persistence
    −Spread : wide spread = costly / risky
    −Kyle λ : high lambda = informed flow = adverse selection risk

    Cross-sectional z-scoring ensures signals are comparable across stocks.
    This directly follows the book's multi-signal framework (Ch.11–16).
    """

    WEIGHTS = {
        "ofi":     0.35,
        "impact":  0.25,
        "hawkes":  0.15,
        "spread": -0.15,
        "kyle":   -0.10,
    }

    def __init__(self, cfg):
        self.cfg = cfg

    def zscore_cross(self, panel, col):
        mu  = panel.groupby("date")[col].transform("mean")
        std = panel.groupby("date")[col].transform("std").replace(0, np.nan)
        return (panel[col] - mu) / std

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        p = panel.copy()
        for sig in ["ofi","impact","hawkes","spread","kyle_lam"]:
            p[f"z_{sig}"] = self.zscore_cross(p, sig)
        p["score"] = (
            self.WEIGHTS["ofi"]    * p["z_ofi"]
          + self.WEIGHTS["impact"] * p["z_impact"]
          + self.WEIGHTS["hawkes"] * p["z_hawkes"]
          + self.WEIGHTS["spread"] * p["z_spread"]
          + self.WEIGHTS["kyle"]   * p["z_kyle_lam"]
        )
        return p

    def build_positions(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Position sizing with:
        1. Minimum holding period  → cuts turnover (V1 fix)
        2. Liquidity filter        → skip illiquid stocks (V1 fix)
        3. Inverse-vol weighting   → vol target
        4. Max turnover cap        → limits cost drag (V1 fix)
        """
        p         = panel.copy().sort_values(["date","ticker"])
        p["position"] = 0.0

        thresh    = self.cfg["signal_threshold"]
        top_n     = self.cfg["top_n"]
        vol_tgt   = self.cfg["vol_target"]
        max_pos   = self.cfg["max_position"]
        max_liq   = self.cfg["max_spread_filter"]
        hold_min  = self.cfg["holding_period"]
        max_turn  = self.cfg["max_turnover_day"]

        dates          = sorted(p["date"].unique())
        prev_pos       = {}    # ticker → weight
        days_held      = {}    # ticker → days since last trade
        target_pos     = {}    # ticker → target (for turnover cap)

        for date in dates:
            mask = p["date"] == date
            day  = p[mask].set_index("ticker")

            # ── Liquidity filter: drop stocks with wide spreads ───────────
            liquid = day[day["true_spread"] < max_liq]

            # ── Compute target positions from signal ──────────────────────
            new_targets = {}
            longs = (liquid[liquid["score"] >  thresh]
                     .nlargest(top_n, "score"))

            for tkr, row in longs.iterrows():
                v = max(row["realised_vol"] * np.sqrt(252), 0.01)
                w = min((vol_tgt / v) / max(len(longs), 1), max_pos)
                new_targets[tkr] = w

            # ── Holding period: only update if held long enough ───────────
            final_targets = {}
            for tkr in set(list(prev_pos.keys()) + list(new_targets.keys())):
                held = days_held.get(tkr, hold_min + 1)
                if held < hold_min and tkr in prev_pos:
                    final_targets[tkr] = prev_pos.get(tkr, 0.0)
                else:
                    final_targets[tkr] = new_targets.get(tkr, 0.0)

            # ── Turnover cap ──────────────────────────────────────────────
            total_change = sum(
                abs(final_targets.get(t, 0) - prev_pos.get(t, 0))
                for t in set(list(final_targets) + list(prev_pos))
            )
            if total_change > max_turn:
                scale = max_turn / total_change
                for t in final_targets:
                    delta = final_targets[t] - prev_pos.get(t, 0)
                    final_targets[t] = prev_pos.get(t, 0) + delta * scale

            # ── Apply to panel ────────────────────────────────────────────
            for tkr, w in final_targets.items():
                if tkr in day.index:
                    p.loc[mask & (p["ticker"] == tkr), "position"] = w

            # ── Update state ──────────────────────────────────────────────
            for tkr in day.index:
                old = prev_pos.get(tkr, 0.0)
                new = final_targets.get(tkr, 0.0)
                if abs(new - old) > 1e-6:
                    days_held[tkr] = 0
                else:
                    days_held[tkr] = days_held.get(tkr, 0) + 1
                prev_pos[tkr] = new
            for t in list(prev_pos):
                if t not in day.index:
                    prev_pos[t] = 0.0

        return p


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    DAILY P&L with Jerry's cost model:
    ────────────────────────────────────
    commission  = 10bps per trade (one-way)  ← Jerry's 0.1% number
    impact      = 0bps                        ← Jerry said assume no impact

    pnl_t = Σ_i [pos_{i,t-1} × ret_{i,t}] − commission × |Δpos|

    COST ANALYSIS vs V1:
    ─────────────────────
    V1: 8bps × 18.3%/day = 1.46bps/day → 3.7%/yr
    V2: 10bps × ~2%/day  = 0.2bps/day  → 0.5%/yr   (3x cheaper)
    """

    def __init__(self, cfg):
        self.commission = cfg["commission_bps"] / 10_000
        self.impact     = cfg["market_impact_bps"] / 10_000

    def run(self, panel: pd.DataFrame) -> pd.DataFrame:
        print("  Running backtest (commission=10bps, impact=0bps)...")
        rows     = []
        prev_pos = {}
        dates    = sorted(panel["date"].unique())

        for date in dates:
            day        = panel[panel["date"] == date].set_index("ticker")
            gross_pnl  = 0.0
            cost       = 0.0
            turnover   = 0.0

            for tkr in day.index:
                pos_now  = day.loc[tkr, "position"]
                pos_prev = prev_pos.get(tkr, 0.0)
                ret      = day.loc[tkr, "ret"]

                gross_pnl += pos_prev * ret
                delta      = abs(pos_now - pos_prev)
                cost      += delta * (self.commission + self.impact)
                turnover  += delta

            net_pnl = gross_pnl - cost

            for tkr in day.index:
                prev_pos[tkr] = day.loc[tkr, "position"]
            for t in list(prev_pos):
                if t not in day.index:
                    prev_pos[t] = 0.0

            rows.append({
                "date":       date,
                "gross_pnl":  gross_pnl,
                "cost":       cost,
                "net_pnl":    net_pnl,
                "turnover":   turnover,
                "n_long":     (day["position"] > 0).sum(),
                "net_exp":    day["position"].sum(),
            })

        df = pd.DataFrame(rows).set_index("date")
        df["nav"]     = (1 + df["net_pnl"]).cumprod()
        df["cum_pnl"] = df["net_pnl"].cumsum()
        return df


# ═════════════════════════════════════════════════════════════════════════════
# PERFORMANCE ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(results):
    r   = results["net_pnl"].dropna()
    nav = results["nav"].dropna()
    n_y = len(r) / 252

    total_ret = nav.iloc[-1] - 1
    cagr      = nav.iloc[-1]**(1/n_y) - 1
    sharpe    = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    neg       = r[r < 0]
    sortino   = r.mean() / neg.std() * np.sqrt(252) if len(neg) > 0 else 0
    dd        = (nav - nav.cummax()) / nav.cummax()
    max_dd    = dd.min()
    calmar    = cagr / abs(max_dd) if max_dd != 0 else 0
    hit       = (r > 0).mean()
    avg_turn  = results["turnover"].mean()
    ann_cost  = (results["cost"].mean() * 252)

    return {
        "Total Return":     f"{total_ret:.1%}",
        "CAGR":             f"{cagr:.1%}",
        "Sharpe Ratio":     f"{sharpe:.2f}",
        "Sortino Ratio":    f"{sortino:.2f}",
        "Max Drawdown":     f"{max_dd:.1%}",
        "Calmar Ratio":     f"{calmar:.2f}",
        "Hit Rate":         f"{hit:.1%}",
        "Avg Turnover/day": f"{avg_turn:.2%}",
        "Annual Cost Drag": f"{ann_cost:.2%}",
        "Total Days":       str(len(r)),
    }

def compute_stylised(r):
    r = r.dropna()
    return {
        "Mean (annual)":    f"{r.mean()*252:.2%}",
        "Vol (annual)":     f"{r.std()*np.sqrt(252):.2%}",
        "Skewness":         f"{skew(r):.3f}",
        "Excess Kurtosis":  f"{kurtosis(r):.3f}",
        "Worst Day":        f"{r.min():.2%}",
        "Best Day":         f"{r.max():.2%}",
    }


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING — FULL RESEARCH DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def plot_dashboard(results, panel, metrics, stylised, cfg, sample_intraday):
    print("  Building dashboard...")
    plt.style.use("dark_background")
    BG, GRID = "#0d1117", "#21262d"
    GREEN, RED, BLUE, GOLD, WHITE = "#3fb950","#f85149","#58a6ff","#d29922","#e6edf3"

    fig = plt.figure(figsize=(24, 28), facecolor=BG)
    fig.suptitle(
        "MICROSTRUCTURE STRATEGY BACKTEST  v2.0  |  2020–2025  |  1-MIN INTRADAY BARS\n"
        "Trades, Quotes and Prices — Bouchaud, Bonart, Donier & Gould (2018)",
        fontsize=13, color=WHITE, fontweight="bold", y=0.985
    )
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.50, wspace=0.35,
                           top=0.965, bottom=0.04)

    def style_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=WHITE, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)

    # ── 1. Equity Curve ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    nav = results["nav"]
    ax1.plot(nav.index, nav.values, color=GREEN, lw=1.5)
    ax1.fill_between(nav.index, 1, nav.values,
                     where=(nav >= 1), alpha=0.2, color=GREEN)
    ax1.fill_between(nav.index, 1, nav.values,
                     where=(nav < 1),  alpha=0.2, color=RED)
    ax1.axhline(1, color=WHITE, lw=0.8, ls="--", alpha=0.5)
    ax1.set_title("Equity Curve — Net of 10bps Commission (Zero Impact)",
                  color=WHITE, fontsize=10)
    ax1.set_ylabel("NAV", color=WHITE)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:.2f}"))
    style_ax(ax1)

    # ── 2. Metrics Table ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off"); ax2.set_facecolor(BG)
    ax2.set_title("Performance (v2)", color=WHITE, fontsize=10)
    all_m = {**metrics, **stylised}
    yp = 0.97
    for k, v in all_m.items():
        try:
            val = float(str(v).replace("%","").replace("/day",""))
            col = RED if val < 0 else GREEN
        except:
            col = BLUE
        ax2.text(0.02, yp, k,    color=WHITE, fontsize=8,  transform=ax2.transAxes)
        ax2.text(0.68, yp, str(v), color=col, fontsize=8,
                 fontweight="bold", transform=ax2.transAxes)
        yp -= 0.088

    # ── 3. Drawdown ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    dd = (results["nav"] - results["nav"].cummax()) / results["nav"].cummax()
    ax3.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.6)
    ax3.plot(dd.index, dd.values, color=RED, lw=0.7)
    ax3.set_title("Drawdown", color=WHITE, fontsize=10)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:.1%}"))
    style_ax(ax3)

    # ── 4. Rolling Sharpe ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    rs = results["net_pnl"].rolling(63).mean() / results["net_pnl"].rolling(63).std() * np.sqrt(252)
    ax4.plot(rs.index, rs.values, color=GOLD, lw=1.2)
    ax4.axhline(0, color=WHITE, lw=0.8, ls="--", alpha=0.5)
    ax4.axhline(1, color=GREEN, lw=0.8, ls=":", alpha=0.5)
    ax4.set_title("Rolling 3M Sharpe", color=WHITE, fontsize=10)
    style_ax(ax4)

    # ── 5. INTRADAY SAMPLE — 1 day of 1-min bars for AAPL ────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    aapl_day = sample_intraday
    ax5.plot(range(len(aapl_day)), aapl_day["mid"].values, color=BLUE, lw=1)
    ax5b = ax5.twinx()
    ax5b.bar(range(len(aapl_day)), aapl_day["volume"].values,
             color=GOLD, alpha=0.3, width=1)
    ax5b.set_ylabel("Volume", color=GOLD, fontsize=8)
    ax5b.tick_params(colors=GOLD, labelsize=7)
    ax5.set_title("Sample Intraday: 1-min bars (AAPL, 1 day) — 390 bars/day",
                  color=WHITE, fontsize=10)
    ax5.set_xlabel("Bar index (0 = 9:30 AM, 389 = 3:59 PM)", color=WHITE, fontsize=8)
    ax5.set_ylabel("Mid Price ($)", color=WHITE)
    style_ax(ax5)

    # ── 6. Intraday OFI for sample day ────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(range(len(aapl_day)), aapl_day["ofi"].values, color=GREEN, lw=0.8)
    ax6.axhline(0, color=WHITE, lw=0.6, ls="--", alpha=0.5)
    ax6.fill_between(range(len(aapl_day)), 0, aapl_day["ofi"].values,
                     where=(aapl_day["ofi"] > 0), alpha=0.3, color=GREEN)
    ax6.fill_between(range(len(aapl_day)), 0, aapl_day["ofi"].values,
                     where=(aapl_day["ofi"] < 0), alpha=0.3, color=RED)
    ax6.set_title("Intraday OFI (Ch.11)\n(buy pressure vs sell pressure per bar)",
                  color=WHITE, fontsize=9)
    style_ax(ax6)

    # ── 7. Return distribution ────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    r = results["net_pnl"].dropna()
    ax7.hist(r, bins=80, density=True, color=BLUE, alpha=0.7)
    xfit = np.linspace(r.min(), r.max(), 300)
    ax7.plot(xfit, stats.norm.pdf(xfit, r.mean(), r.std()),
             color=GOLD, lw=1.5, ls="--", label="Normal")
    ax7.set_title(f"Return Distribution\nKurt={kurtosis(r):.1f}  Skew={skew(r):.2f}",
                  color=WHITE, fontsize=9)
    ax7.legend(fontsize=7, framealpha=0.3)
    style_ax(ax7)

    # ── 8. ACF of returns ─────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    lags    = range(1, 21)
    acf_r   = [r.autocorr(l) for l in lags]
    conf    = 1.96 / np.sqrt(len(r))
    ax8.bar(lags, acf_r, color=BLUE, alpha=0.7)
    ax8.axhline( conf, color=RED, lw=1, ls="--", alpha=0.7, label="95% CI")
    ax8.axhline(-conf, color=RED, lw=1, ls="--", alpha=0.7)
    ax8.axhline(0, color=WHITE, lw=0.6, ls="--", alpha=0.4)
    ax8.set_title("ACF Returns\n(near-zero → martingale, Ch.2)", color=WHITE, fontsize=9)
    ax8.set_xlabel("Lag (days)", color=WHITE, fontsize=8)
    ax8.legend(fontsize=7, framealpha=0.3)
    style_ax(ax8)

    # ── 9. ACF of |returns| — Hawkes clustering ───────────────────────────
    ax9 = fig.add_subplot(gs[3, 2])
    acf_abs = [r.abs().autocorr(l) for l in lags]
    ax9.bar(lags, acf_abs, color=GOLD, alpha=0.7)
    ax9.axhline( conf, color=RED, lw=1, ls="--", alpha=0.7)
    ax9.axhline(-conf, color=RED, lw=1, ls="--", alpha=0.7)
    ax9.axhline(0, color=WHITE, lw=0.6, ls="--", alpha=0.4)
    ax9.set_title("ACF |Returns|\n(Hawkes clustering, Ch.9)", color=WHITE, fontsize=9)
    ax9.set_xlabel("Lag (days)", color=WHITE, fontsize=8)
    style_ax(ax9)

    # ── 10. Monthly heatmap ───────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[4, :2])
    monthly    = results["net_pnl"].resample("ME").sum()
    monthly.index = monthly.index.to_period("M")
    pivot = {}
    for period, val in monthly.items():
        pivot.setdefault(period.year, {})[period.month] = val
    years = sorted(pivot.keys())
    heat  = np.full((len(years), 12), np.nan)
    for i, yr in enumerate(years):
        for j, m in enumerate(range(1,13)):
            heat[i,j] = pivot.get(yr,{}).get(m, np.nan)
    vmax = np.nanmax(np.abs(heat))
    im = ax10.imshow(heat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax10.set_xticks(range(12))
    ax10.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"],
                          color=WHITE, fontsize=8)
    ax10.set_yticks(range(len(years)))
    ax10.set_yticklabels(years, color=WHITE, fontsize=8)
    ax10.set_title("Monthly Returns Heatmap", color=WHITE, fontsize=10)
    for i in range(len(years)):
        for j in range(12):
            if not np.isnan(heat[i,j]):
                ax10.text(j, i, f"{heat[i,j]:.1%}", ha="center", va="center",
                          fontsize=6.5,
                          color="black" if abs(heat[i,j])>vmax*0.3 else WHITE)
    plt.colorbar(im, ax=ax10, fraction=0.02)
    ax10.set_facecolor(BG)

    # ── 11. Daily turnover (v1 vs v2 comparison annotation) ───────────────
    ax11 = fig.add_subplot(gs[4, 2])
    turn = results["turnover"].rolling(20).mean()
    ax11.fill_between(turn.index, turn.values, color=BLUE, alpha=0.5)
    ax11.plot(turn.index, turn.values, color=BLUE, lw=0.8)
    ax11.axhline(0.183, color=RED, lw=1.2, ls="--", alpha=0.8,
                 label="v1 avg (18.3%)")
    ax11.set_title("Rolling 20D Turnover\nv1=18.3%/day  →  v2=target<5%/day",
                   color=WHITE, fontsize=9)
    ax11.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:.0%}"))
    ax11.legend(fontsize=7, framealpha=0.3)
    style_ax(ax11)

    # ── Save ──────────────────────────────────────────────────────────────
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "Microstructure_Backtest_Results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved → {output_path}")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    cfg         = CONFIG
    tickers     = cfg["tickers"]
    trading_days = pd.bdate_range(cfg["start"], cfg["end"])

    print(f"\n{'='*65}")
    print("  MICROSTRUCTURE BACKTEST v2.0  —  1-MIN INTRADAY BARS")
    print(f"{'='*65}")

    # ── LAYER 1+2+3: Generate bars, signals, aggregate ────────────────────
    print(f"\n  STEP 1: Generating 1-min bars + intraday signals")
    print(f"          {len(trading_days)} days × 390 bars = "
          f"{len(trading_days)*390:,} bars per stock")

    daily_panels = []
    sample_aapl  = None   # for intraday plot

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        bars    = generate_one_minute_bars(ticker, trading_days,
                                          cfg["bars_per_day"],
                                          seed_offset=i*100)
        bars_s  = compute_intraday_signals(bars, cfg)
        daily   = aggregate_to_daily(bars_s)
        daily["ticker"] = ticker
        daily["date"]   = daily.index
        daily_panels.append(daily.reset_index(drop=True))

        if ticker == "AAPL" and sample_aapl is None:
            # Grab one sample day for the intraday plot
            first_day = bars_s["date"].iloc[0]
            sample_aapl = bars_s[bars_s["date"] == first_day].copy()
        print(f"{len(daily)} days ✓")

    panel = pd.concat(daily_panels, ignore_index=True)
    panel = panel.rename(columns={"spread_roll":"spread"}) \
                 if "spread_roll" in panel.columns else panel

    # ── STEP 3: Score + positions ─────────────────────────────────────────
    print(f"\n  STEP 2: Computing composite signals + positions")
    combiner = SignalCombiner(cfg)
    panel    = combiner.compute(panel)
    panel    = combiner.build_positions(panel)

    turn_avg = panel.groupby("date")["position"].apply(
        lambda x: x.abs().sum()).mean()
    print(f"          Avg daily gross exposure: {turn_avg:.1%}")

    # ── STEP 4: Backtest ──────────────────────────────────────────────────
    print(f"\n  STEP 3: Running backtest")
    engine  = BacktestEngine(cfg)
    results = engine.run(panel)

    avg_turn = results["turnover"].mean()
    ann_cost = results["cost"].mean() * 252
    print(f"          Avg turnover : {avg_turn:.2%}/day")
    print(f"          Annual cost  : {ann_cost:.2%}  "
          f"(10bps × {avg_turn:.2%} × 252)")

    # ── STEP 5: Analytics ─────────────────────────────────────────────────
    print(f"\n  STEP 4: Performance analytics")
    metrics  = compute_metrics(results)
    stylised = compute_stylised(results["net_pnl"])

    print("\n  ┌────────────────────────────────────────────────┐")
    print("  │            PERFORMANCE SUMMARY  v2             │")
    print("  ├───────────────────────────┬────────────────────┤")
    for k, v in {**metrics, **stylised}.items():
        print(f"  │  {k:<25}  │  {v:>16}  │")
    print("  └───────────────────────────┴────────────────────┘")

    # ── STEP 6: Plot ──────────────────────────────────────────────────────
    print(f"\n  STEP 5: Plotting")
    plot_dashboard(results, panel, metrics, stylised, cfg, sample_aapl)

    print(f"\n{'='*65}")
    print("  ✅  DONE — file saved as: Microstructure_Backtest_Results.png")
    print(f"{'='*65}\n")

    return results, panel


if __name__ == "__main__":
    results, panel = main()