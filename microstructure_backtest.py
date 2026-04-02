"""
=============================================================================
MICROSTRUCTURE STRATEGY BACKTEST  v3.0
"Trades, Quotes and Prices" — Bouchaud, Bonart, Donier & Gould (2018)
Cambridge University Press
=============================================================================

KEY FIXES IN v3 (vs v2):
──────────────────────────
  BUG 1 FIXED: Position write-back was broken in v2 — positions were
               computed but never correctly applied to the panel.
               Result: NAV diverged wildly after Sep 2021. NOW FIXED.

  BUG 2 FIXED: Daily return summed bar-level returns incorrectly.
               Cross-day bar returns were not zeroed out.
               NOW: bar-0 return = 0 each day; daily ret = Σ(390 bars).

  BUG 3 FIXED: Holding-period dict mutated mid-loop causing state corruption.
               NOW: clean two-pass (compute target → merge → apply).

NEW: ANTHROPIC API INTEGRATION
────────────────────────────────
  After backtest completes, Claude analyses results in the language of
  the book — citing chapters, equations, empirical findings.

SIGNALS (faithful to the book):
─────────────────────────────────
  OFI       Ch.11 — (V_buy − V_sell) / V_total  [Eq. 11.x]
  Hawkes    Ch.9  — ACF(|r_t|,lag=1) rolling
  Spread    Ch.16 — Roll (1984): 2√(−Cov(Δp_t, Δp_{t-1}))
  Kyle λ    Ch.15 — Amihud: |r| / dollar_volume
  Impact    Ch.12 — sign × √(V/ADV)

DEPENDENCIES: numpy pandas matplotlib requests
RUN: python3 Microstructure_Backtest_TQP_Replication.py
=============================================================================
"""

import os, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── Lightweight stats (no scipy) ──────────────────────────────────────────────
class stats:
    class norm:
        @staticmethod
        def pdf(x, mu, sigma):
            x = np.asarray(x, float)
            return np.exp(-0.5*((x-mu)/sigma)**2) / (sigma*np.sqrt(2*np.pi))

def skew(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    m, s = x.mean(), x.std()
    return float(np.mean(((x-m)/s)**3)) if s > 0 else 0.0

def kurtosis(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    m, s = x.mean(), x.std()
    return float(np.mean(((x-m)/s)**4) - 3) if s > 0 else 0.0

# ═════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "tickers":   ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "start":     "2020-01-01",
    "end":       "2025-01-01",
    "bars_per_day": 390,

    "ofi_bars":    20,
    "hawkes_bars": 60,
    "spread_bars": 30,
    "kyle_bars":   60,

    "signal_threshold": 0.5,
    "top_n":            3,

    "commission_bps":    10.0,
    "market_impact_bps":  0.0,

    "holding_period_days": 3,
    "max_daily_turnover":  0.03,
    "max_weight":          0.25,

    "vol_target":   0.15,
    "min_daily_vol": 0.004,

    "anthropic_model": "claude-sonnet-4-20250514",
}

STOCK_PARAMS = {
    "AAPL":  dict(mu=0.28, sigma=0.0175, spread_bps=1.5,  jump_lam=0.008, p0=75),
    "MSFT":  dict(mu=0.25, sigma=0.0160, spread_bps=1.2,  jump_lam=0.007, p0=160),
    "GOOGL": dict(mu=0.22, sigma=0.0190, spread_bps=2.0,  jump_lam=0.009, p0=67),
    "AMZN":  dict(mu=0.20, sigma=0.0210, spread_bps=2.5,  jump_lam=0.010, p0=94),
    "META":  dict(mu=0.18, sigma=0.0260, spread_bps=3.0,  jump_lam=0.014, p0=208),
    "NVDA":  dict(mu=0.40, sigma=0.0290, spread_bps=2.0,  jump_lam=0.013, p0=59),
    "TSLA":  dict(mu=0.22, sigma=0.0380, spread_bps=4.5,  jump_lam=0.018, p0=86),
}

# ═════════════════════════════════════════════════════════════════════════════
# LAYER 1 — 1-MIN BAR GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def generate_bars(ticker, trading_days, bars_per_day, seed=0):
    np.random.seed(42 + seed)
    p      = STOCK_PARAMS[ticker]
    n_days = len(trading_days)
    N      = n_days * bars_per_day
    dt     = 1.0 / (252 * bars_per_day)

    # GARCH(1,1) [Ch.9]
    bar_var = p["sigma"]**2 / bars_per_day
    omega   = bar_var * 0.05;  alpha_g = 0.10;  beta_g = 0.85
    h   = np.zeros(N);  h[0] = bar_var
    eps = np.zeros(N)
    z   = np.random.standard_t(df=4, size=N)   # fat tails [Ch.2.2]
    for t in range(1, N):
        h[t]   = omega + alpha_g * eps[t-1]**2 + beta_g * h[t-1]
        eps[t] = np.sqrt(max(h[t], 1e-14)) * z[t]

    # Intraday U-shape [Ch.4.2]
    bar_idx   = np.tile(np.arange(bars_per_day), n_days)
    norm_time = bar_idx / (bars_per_day - 1)
    u_shape   = 1.8 - np.sin(np.pi * norm_time) * 0.8

    # Hawkes order signs [Ch.9, Ch.10]
    signs = np.zeros(N); signs[0] = 1.0
    rho   = 0.30
    ru    = np.random.rand(N)
    for t in range(1, N):
        signs[t] = 1.0 if ru[t] < 0.5 + 0.5 * rho * signs[t-1] else -1.0

    # Jumps [Ch.2.2]
    jumps = np.zeros(N)
    jmask = np.random.rand(N) < p["jump_lam"] / bars_per_day
    jumps[jmask] = np.random.normal(0, p["sigma"] * 2.5, jmask.sum())

    # Price path
    drift   = p["mu"] * dt
    bar_ret = (drift + eps + jumps).clip(-0.04, 0.04)
    mid     = np.exp(np.log(p["p0"]) + np.cumsum(bar_ret))

    # Bid-ask [Ch.16]
    half_s  = mid * (p["spread_bps"] / 20_000)
    close   = np.maximum(mid + signs * half_s, 0.01)
    open_   = np.maximum(np.roll(close, 1), 0.01); open_[0] = p["p0"]

    rng  = mid * np.sqrt(np.maximum(h, 1e-14)) * 3.0
    high = np.maximum(open_, close) + rng * np.random.uniform(0.1, 0.5, N)
    low  = np.maximum(np.minimum(open_, close) - rng * np.random.uniform(0.1, 0.5, N), 0.01)

    # Volume [Ch.4.5]
    base_v = 800_000 / max(p["p0"], 1)
    vol_sc = np.sqrt(np.maximum(h, 1e-14)) / np.sqrt(bar_var + 1e-14)
    volume = (base_v * vol_sc * u_shape * np.random.lognormal(0, 0.45, N)).clip(500).astype(np.int64)

    # Build timestamps
    ts = []
    for day in trading_days:
        t0 = pd.Timestamp(day.year, day.month, day.day, 9, 30)
        ts += [t0 + pd.Timedelta(minutes=b) for b in range(bars_per_day)]

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "mid": mid, "sign": signs,
        "h_var": h, "half_spread": half_s,
    }, index=pd.DatetimeIndex(ts))

    # Log returns — zero out day boundaries (KEY FIX)
    df["ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0).clip(-0.04, 0.04)
    # Reset return at the first bar of each day to zero
    first_bars = [i * bars_per_day for i in range(n_days)]
    df.iloc[first_bars, df.columns.get_loc("ret")] = 0.0

    df["date"] = pd.to_datetime(df.index.date)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 2 — INTRADAY SIGNALS
# ═════════════════════════════════════════════════════════════════════════════

def compute_signals(bars, cfg):
    b   = bars.copy()
    Wo  = cfg["ofi_bars"]
    Wh  = cfg["hawkes_bars"]
    Ws  = cfg["spread_bars"]
    Wk  = cfg["kyle_bars"]

    # OFI [Ch.11.4]
    sv       = b["sign"] * b["volume"]
    b["ofi"] = (sv.rolling(Wo).sum() / (b["volume"].rolling(Wo).sum() + 1)).clip(-1, 1)

    # Square-root impact [Ch.12.3]
    adv          = b["volume"].rolling(Wo).mean()
    b["impact"]  = b["sign"] * np.sqrt((b["volume"] / (adv + 1)).clip(0, 10))

    # Roll spread [Ch.16]
    cov_r        = (b["ret"] * b["ret"].shift(1)).rolling(Ws).mean()
    b["spread"]  = (2 * np.sqrt((-cov_r).clip(lower=0))).rolling(Ws).mean()

    # Hawkes [Ch.9.2]
    abs_r        = b["ret"].abs()
    num          = (abs_r * abs_r.shift(1)).rolling(Wh).mean()
    den          = abs_r.rolling(Wh).var() + 1e-10
    b["hawkes"]  = (num / den).clip(0, 5)

    # Kyle lambda [Ch.15.2]
    dflow        = (b["sign"] * b["volume"] * b["mid"]).abs() / 1e6 + 1e-6
    b["kyle"]    = (b["ret"].abs() / dflow).clip(0, 100).rolling(Wk).mean()

    return b.dropna()


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 3 — DAILY AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_daily(b):
    grp = b.groupby("date")
    agg = pd.DataFrame({
        "ofi":          grp["ofi"].apply(lambda x: x.iloc[-60:].mean()),
        "impact":       grp["impact"].mean(),
        "spread":       grp["spread"].mean(),
        "hawkes":       grp["hawkes"].max(),
        "kyle":         grp["kyle"].mean(),
        "daily_ret":    grp["ret"].sum().clip(-0.12, 0.12),
        "realised_vol": grp["ret"].std() * np.sqrt(390),
        "true_spread":  grp["half_spread"].mean() / grp["mid"].mean() * 2,
        "close":        grp["close"].last(),
    })
    agg.index = pd.to_datetime(agg.index)
    return agg.dropna()


# ═════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE
# ═════════════════════════════════════════════════════════════════════════════

def build_score(panel):
    W = {"ofi": 0.35, "impact": 0.25, "hawkes": 0.15, "spread": -0.15, "kyle": -0.10}
    p = panel.copy()
    for sig in W:
        mu  = p.groupby("date")[sig].transform("mean")
        std = p.groupby("date")[sig].transform("std").replace(0, np.nan)
        p[f"z_{sig}"] = ((p[sig] - mu) / std).clip(-3, 3)
    p["score"] = sum(W[s] * p[f"z_{s}"] for s in W)
    return p


# ═════════════════════════════════════════════════════════════════════════════
# POSITION BUILDER — v3 FIXED
# ═════════════════════════════════════════════════════════════════════════════

def build_positions(panel, cfg):
    """
    Clean implementation: compute targets per day, merge back to panel.
    Avoids the in-place mutation bugs of v2.
    """
    p      = panel.sort_values(["date","ticker"]).reset_index(drop=True)
    dates  = sorted(p["date"].unique())

    thresh    = cfg["signal_threshold"]
    top_n     = cfg["top_n"]
    vol_tgt   = cfg["vol_target"]
    max_w     = cfg["max_weight"]
    hold_days = cfg["holding_period_days"]
    max_turn  = cfg["max_daily_turnover"]
    min_vol   = cfg["min_daily_vol"]

    pos_dict   = {}   # ticker → weight
    days_since = {}   # ticker → days since last change

    all_records = []

    for date in dates:
        day = p[p["date"] == date].set_index("ticker")

        # Liquidity filter
        liquid = day[day["realised_vol"] > min_vol]

        # Raw targets
        candidates = liquid[liquid["score"] > thresh].nlargest(top_n, "score")
        raw = {}
        n   = max(len(candidates), 1)
        for tkr, row in candidates.iterrows():
            rv = max(row["realised_vol"] * np.sqrt(252), 0.05)
            raw[tkr] = min(vol_tgt / rv / n, max_w)

        # Holding period
        targets = {}
        for tkr in set(list(pos_dict.keys()) + list(raw.keys())):
            old  = pos_dict.get(tkr, 0.0)
            new  = raw.get(tkr, 0.0)
            held = days_since.get(tkr, 999)
            targets[tkr] = old if (held < hold_days and old > 0) else new

        # Turnover cap
        total_chg = sum(abs(targets.get(t, 0) - pos_dict.get(t, 0))
                        for t in set(list(targets) + list(pos_dict)))
        if total_chg > max_turn and total_chg > 0:
            sc = max_turn / total_chg
            targets = {t: pos_dict.get(t, 0) + (targets[t] - pos_dict.get(t, 0)) * sc
                       for t in targets}

        # Update state
        for tkr in set(list(targets.keys()) + list(pos_dict.keys())):
            old = pos_dict.get(tkr, 0.0)
            new = targets.get(tkr, 0.0)
            days_since[tkr] = 0 if abs(new - old) > 1e-6 else days_since.get(tkr, 0) + 1
            pos_dict[tkr]   = new

        # Record for ALL tickers on this date
        for tkr in day.index:
            all_records.append({"date": date, "ticker": tkr,
                                 "position": targets.get(tkr, 0.0)})

    pos_df = pd.DataFrame(all_records)
    p = p.merge(pos_df, on=["date","ticker"], how="left")
    p["position"] = p["position"].fillna(0.0)
    return p


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE — v3
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest(panel, cfg):
    """
    P&L_t = Σ_i [pos_{i,t-1} × ret_{i,t}] - costs_t

    Key: use position from YESTERDAY to earn TODAY's return.
    """
    comm = cfg["commission_bps"] / 10_000
    imp  = cfg["market_impact_bps"] / 10_000

    dates    = sorted(panel["date"].unique())
    prev_pos = {}
    rows     = []

    for date in dates:
        day = panel[panel["date"] == date].set_index("ticker")

        gross = 0.0; cost = 0.0; turn = 0.0

        for tkr in day.index:
            pos_now  = day.loc[tkr, "position"]
            pos_prev = prev_pos.get(tkr, 0.0)
            ret      = day.loc[tkr, "daily_ret"]

            if np.isfinite(ret):
                gross += pos_prev * ret

            delta = abs(pos_now - pos_prev)
            cost += delta * (comm + imp)
            turn += delta

        for tkr in day.index:
            prev_pos[tkr] = day.loc[tkr, "position"]
        for t in list(prev_pos):
            if t not in day.index:
                prev_pos[t] = 0.0

        rows.append({
            "date":      date,
            "gross_pnl": gross,
            "cost":      cost,
            "net_pnl":   gross - cost,
            "turnover":  turn,
            "n_long":    (day["position"] > 0).sum(),
        })

    res           = pd.DataFrame(rows).set_index("date")
    res["nav"]    = (1 + res["net_pnl"]).cumprod()
    res["cum_pnl"]= res["net_pnl"].cumsum()
    return res


# ═════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════

def metrics(res):
    r   = res["net_pnl"].dropna()
    nav = res["nav"].dropna()
    n_y = len(r) / 252
    tot = nav.iloc[-1] - 1
    cagr= nav.iloc[-1]**(1/n_y) - 1
    sh  = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    neg = r[r < 0]
    so  = r.mean() / neg.std() * np.sqrt(252) if len(neg) > 1 else 0
    dd  = (nav - nav.cummax()) / nav.cummax()
    mdd = dd.min()
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return {
        "Total Return":   f"{tot:.2%}",
        "CAGR":           f"{cagr:.2%}",
        "Sharpe Ratio":   f"{sh:.3f}",
        "Sortino Ratio":  f"{so:.3f}",
        "Max Drawdown":   f"{mdd:.2%}",
        "Calmar Ratio":   f"{cal:.3f}",
        "Hit Rate":       f"{(r>0).mean():.1%}",
        "Avg Turnover":   f"{res['turnover'].mean():.3%}/day",
        "Ann. Cost Drag": f"{res['cost'].mean()*252:.3%}",
        "Trading Days":   f"{len(r)}",
    }

def stylised_facts(r):
    r = r.dropna()
    return {
        "Ann. Mean":       f"{r.mean()*252:.2%}",
        "Ann. Vol":        f"{r.std()*np.sqrt(252):.2%}",
        "Skewness":        f"{skew(r):.3f}",
        "Excess Kurtosis": f"{kurtosis(r):.3f}",
        "Worst Day":       f"{r.min():.2%}",
        "Best Day":        f"{r.max():.2%}",
    }


# ═════════════════════════════════════════════════════════════════════════════
# ANTHROPIC API
# ═════════════════════════════════════════════════════════════════════════════

def get_ai_commentary(met, sty, cfg):
    if not HAS_REQUESTS:
        return "[ pip install requests to enable AI commentary ]"

    stats_str = "\n".join(f"  {k}: {v}" for k, v in {**met, **sty}.items())
    prompt = f"""You are a senior quant researcher reviewing a microstructure backtest.

Strategy based on: "Trades, Quotes and Prices" (Bouchaud, Bonart, Donier & Gould, 2018)
Signals: OFI [Ch.11], Square-root impact [Ch.12], Hawkes [Ch.9], Roll spread [Ch.16], Kyle lambda [Ch.15]
Data: Synthetic 1-min NASDAQ bars, 7 stocks, 2020–2025
Costs: 10bps commission, zero market impact

RESULTS:
{stats_str}

Write a sharp 150-word research commentary:
- What these numbers say about the strategy
- Reference at least 2 specific chapters from the book
- Why this result is theoretically expected
- One concrete improvement idea from the book's framework"""

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={"model": cfg["anthropic_model"], "max_tokens": 400,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        d = r.json()
        if "content" in d:
            return d["content"][0]["text"]
        return f"[ API response: {str(d)[:150]} ]"
    except Exception as e:
        return f"[ API error: {e} ]"


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD PLOT
# ═════════════════════════════════════════════════════════════════════════════

def plot_all(res, panel, met, sty, cfg, sample_bars, ai_text):
    BG, GRID = "#0d1117", "#1c2128"
    G, R, B, Y, W, GR = "#3fb950","#f85149","#58a6ff","#d29922","#e6edf3","#8b949e"

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(26, 32), facecolor=BG)
    fig.suptitle(
        "MICROSTRUCTURE STRATEGY BACKTEST  v3.0  ·  1-MIN INTRADAY BARS  ·  2020–2025\n"
        "\"Trades, Quotes and Prices\" — Bouchaud, Bonart, Donier & Gould (Cambridge, 2018)",
        fontsize=13, color=W, fontweight="bold", y=0.993, fontfamily="monospace"
    )

    gs = gridspec.GridSpec(5, 3, figure=fig,
                           hspace=0.55, wspace=0.35,
                           top=0.977, bottom=0.04,
                           left=0.07, right=0.97)

    def sax(ax, xl="", yl=""):
        ax.set_facecolor(BG)
        ax.tick_params(colors=GR, labelsize=8)
        ax.xaxis.label.set_color(GR); ax.yaxis.label.set_color(GR)
        if xl: ax.set_xlabel(xl, fontsize=8)
        if yl: ax.set_ylabel(yl, fontsize=8)
        for s in ax.spines.values(): s.set_color(GRID)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.6)

    yr_lines = lambda ax: [ax.axvline(pd.Timestamp(f"{y}-01-01"),
                           color=GRID, lw=0.6, ls=":") for y in [2021,2022,2023,2024]]

    # ── 1. Equity curve ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    nav = res["nav"]
    ax1.plot(nav.index, nav, color=G, lw=1.8, zorder=3)
    ax1.fill_between(nav.index, 1, nav.values,
                     where=(nav.values >= 1), alpha=0.18, color=G, zorder=2)
    ax1.fill_between(nav.index, 1, nav.values,
                     where=(nav.values <  1), alpha=0.18, color=R, zorder=2)
    ax1.axhline(1, color=GR, lw=0.8, ls="--", alpha=0.6)
    ax1.set_title("NAV  ·  Net of 10bps Commission  ·  Zero Impact  [Ch.17]",
                  color=W, fontsize=10, pad=6)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:.3f}"))
    yr_lines(ax1); sax(ax1, yl="NAV (start=1.000)")

    # ── 2. Metrics table ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(BG); ax2.axis("off")
    ax2.set_title("Performance Summary", color=W, fontsize=10, pad=6)
    yp = 0.97
    for k, v in {**met, **sty}.items():
        try:
            val = float(str(v).replace("%","").replace("/day",""))
            col = R if val < 0 else (G if val > 0 else GR)
        except: col = B
        ax2.text(0.02, yp, k,  color=GR, fontsize=8.5, transform=ax2.transAxes,
                 fontfamily="monospace")
        ax2.text(0.65, yp, v,  color=col, fontsize=8.5, fontweight="bold",
                 transform=ax2.transAxes, fontfamily="monospace")
        yp -= 0.082

    # ── 3. Drawdown ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    dd  = (res["nav"] - res["nav"].cummax()) / res["nav"].cummax()
    ax3.fill_between(dd.index, dd.values, 0, color=R, alpha=0.65, zorder=2)
    ax3.plot(dd.index, dd.values, color=R, lw=0.8, zorder=3)
    ax3.axhline(0, color=GR, lw=0.6, ls="--", alpha=0.5)
    ax3.set_title("Drawdown from Peak", color=W, fontsize=10, pad=6)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:.1%}"))
    yr_lines(ax3); sax(ax3, yl="Drawdown")

    # ── 4. Rolling Sharpe ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    rs  = (res["net_pnl"].rolling(63).mean() /
           res["net_pnl"].rolling(63).std() * np.sqrt(252))
    ax4.plot(rs.index, rs.values, color=Y, lw=1.2, zorder=3)
    ax4.fill_between(rs.index, 0, rs.values,
                     where=(rs.values > 0), alpha=0.2, color=G)
    ax4.fill_between(rs.index, 0, rs.values,
                     where=(rs.values < 0), alpha=0.2, color=R)
    ax4.axhline(0, color=GR, lw=0.8, ls="--", alpha=0.5)
    ax4.axhline(1, color=G,  lw=0.6, ls=":",  alpha=0.5, label="SR=1")
    ax4.set_title("Rolling 63-Day Sharpe Ratio", color=W, fontsize=10, pad=6)
    ax4.legend(fontsize=7, framealpha=0.3); sax(ax4)

    # ── 5. Sample intraday bars ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    sb  = sample_bars.reset_index(drop=True)
    ax5.plot(sb.index, sb["mid"].values, color=B, lw=1.2, label="Mid price")
    ax5.fill_between(sb.index,
                     (sb["mid"] - sb["half_spread"]).values,
                     (sb["mid"] + sb["half_spread"]).values,
                     alpha=0.25, color=Y, label="Bid-ask spread [Ch.16]")
    ax5v = ax5.twinx()
    ax5v.bar(sb.index, sb["volume"].values, color=GR, alpha=0.2, width=1)
    ax5v.set_ylabel("Volume", color=GR, fontsize=7)
    ax5v.tick_params(colors=GR, labelsize=7)
    ax5.set_title("AAPL — 1-Min Bars (Day 1)  ·  390 bars/day  [Ch.4]",
                  color=W, fontsize=10, pad=6)
    ax5.legend(fontsize=7, framealpha=0.3)
    ax5.set_facecolor(BG); ax5.tick_params(colors=GR, labelsize=8)
    for s in ax5.spines.values(): s.set_color(GRID)
    ax5.set_xlabel("Bar index (0=9:30  →  389=3:59 PM)", fontsize=8, color=GR)
    ax5.set_ylabel("Price ($)", fontsize=8, color=GR)

    # ── 6. Intraday OFI ───────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    if "ofi" in sb.columns:
        ofi = sb["ofi"].fillna(0).values
        ax6.plot(sb.index, ofi, color=G, lw=0.9, zorder=3)
        ax6.fill_between(sb.index, 0, ofi, where=(ofi > 0), alpha=0.3, color=G)
        ax6.fill_between(sb.index, 0, ofi, where=(ofi < 0), alpha=0.3, color=R)
    ax6.axhline(0, color=GR, lw=0.6, ls="--")
    ax6.set_title("Intraday OFI  [Ch.11.4]\n(buy−sell pressure per 1-min bar)",
                  color=W, fontsize=9, pad=4)
    sax(ax6, yl="OFI")

    # ── 7. Return distribution ────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    r   = res["net_pnl"].dropna()
    ax7.hist(r, bins=90, density=True, color=B, alpha=0.72, zorder=2, label="Empirical")
    xf  = np.linspace(r.min(), r.max(), 400)
    ax7.plot(xf, stats.norm.pdf(xf, r.mean(), r.std()),
             color=Y, lw=1.8, ls="--", label="Gaussian [Ch.2.1]", zorder=3)
    ax7.set_title(f"Return Distribution  [Ch.2.2]\nKurt={kurtosis(r):.2f}  Skew={skew(r):.3f}",
                  color=W, fontsize=9, pad=4)
    ax7.legend(fontsize=7, framealpha=0.3)
    sax(ax7, xl="Daily P&L", yl="Density")

    # ── 8. ACF of returns ─────────────────────────────────────────────────
    ax8  = fig.add_subplot(gs[3, 1])
    lags = list(range(1, 22))
    acfr = [r.autocorr(l) for l in lags]
    conf = 1.96 / np.sqrt(len(r))
    ax8.bar(lags, acfr, color=B, alpha=0.75, width=0.6, zorder=2)
    ax8.axhline( conf, color=R, lw=1, ls="--", alpha=0.8, label="95% CI")
    ax8.axhline(-conf, color=R, lw=1, ls="--", alpha=0.8)
    ax8.axhline(0, color=GR, lw=0.5, ls="--", alpha=0.5)
    ax8.set_title("ACF Returns  [Ch.2.1.2]\n(near-zero → martingale)",
                  color=W, fontsize=9, pad=4)
    ax8.legend(fontsize=7, framealpha=0.3)
    sax(ax8, xl="Lag (days)", yl="ACF(r)")

    # ── 9. ACF |returns| — Hawkes ─────────────────────────────────────────
    ax9    = fig.add_subplot(gs[3, 2])
    acfabs = [r.abs().autocorr(l) for l in lags]
    ax9.bar(lags, acfabs, color=Y, alpha=0.75, width=0.6, zorder=2)
    ax9.axhline( conf, color=R, lw=1, ls="--", alpha=0.8)
    ax9.axhline(-conf, color=R, lw=1, ls="--", alpha=0.8)
    ax9.axhline(0, color=GR, lw=0.5, ls="--", alpha=0.5)
    ax9.set_title("ACF |Returns|  [Ch.9]\nHawkes volatility clustering",
                  color=W, fontsize=9, pad=4)
    sax(ax9, xl="Lag (days)", yl="ACF(|r|)")

    # ── 10. Monthly heatmap ───────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[4, :2])
    mo   = res["net_pnl"].resample("ME").sum()
    mo.index = mo.index.to_period("M")
    piv  = {}
    for per, val in mo.items():
        piv.setdefault(per.year, {})[per.month] = val
    years = sorted(piv.keys())
    heat  = np.full((len(years), 12), np.nan)
    for i, yr in enumerate(years):
        for j, m in enumerate(range(1,13)):
            heat[i, j] = piv.get(yr, {}).get(m, np.nan)
    vmax = max(np.nanmax(np.abs(heat)), 0.001)
    im   = ax10.imshow(heat, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                       aspect="auto", interpolation="nearest")
    ax10.set_xticks(range(12))
    ax10.set_xticklabels(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        color=W, fontsize=8)
    ax10.set_yticks(range(len(years)))
    ax10.set_yticklabels(years, color=W, fontsize=8)
    ax10.set_title("Monthly Returns Heatmap", color=W, fontsize=10, pad=6)
    for i in range(len(years)):
        for j in range(12):
            if not np.isnan(heat[i, j]):
                tc = "black" if abs(heat[i,j]) > vmax * 0.4 else W
                ax10.text(j, i, f"{heat[i,j]:.1%}", ha="center", va="center",
                          fontsize=6.5, color=tc, fontweight="bold")
    cb = plt.colorbar(im, ax=ax10, fraction=0.018, pad=0.02)
    cb.ax.tick_params(colors=GR, labelsize=7)
    ax10.set_facecolor(BG)

    # ── 11. Turnover ──────────────────────────────────────────────────────
    ax11 = fig.add_subplot(gs[4, 2])
    turn = res["turnover"].rolling(20).mean() * 100
    ax11.fill_between(turn.index, turn.values, color=B, alpha=0.4)
    ax11.plot(turn.index, turn.values, color=B, lw=0.9)
    ax11.axhline(18.3, color=R, lw=1.2, ls="--", alpha=0.8, label="v1 (18.3%)")
    ax11.axhline(cfg["max_daily_turnover"]*100, color=G, lw=1, ls=":", alpha=0.8,
                 label=f"cap {cfg['max_daily_turnover']*100:.0f}%")
    ax11.set_title("20D Rolling Turnover  [Ch.17]\nv1=18.3%  →  v3 capped",
                   color=W, fontsize=9, pad=4)
    ax11.legend(fontsize=7, framealpha=0.3)
    sax(ax11, yl="Turnover %/day")

    # ── AI commentary ─────────────────────────────────────────────────────
    wrapped = ai_text.replace("\n", "  ")
    if len(wrapped) > 600:
        wrapped = wrapped[:597] + "..."
    fig.text(0.07, 0.006,
             "🤖 Claude API Commentary:  " + wrapped,
             color=GR, fontsize=6.5, fontfamily="monospace",
             verticalalignment="bottom", wrap=True,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                       edgecolor=GRID, alpha=0.95))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out        = os.path.join(script_dir, "Microstructure_Backtest_Results.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"  ✅ Saved → {out}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    cfg          = CONFIG
    trading_days = pd.bdate_range(cfg["start"], cfg["end"])

    print(f"\n{'='*65}")
    print("  MICROSTRUCTURE BACKTEST v3.0 — Bouchaud et al. (2018)")
    print(f"  {cfg['start']} → {cfg['end']}  |  1-min bars  |  {len(cfg['tickers'])} stocks")
    print(f"{'='*65}")

    # Step 1: Data
    print(f"\n  [1/5]  Generating 1-min bars + intraday signals")
    print(f"         {len(trading_days)} days × 390 bars = {len(trading_days)*390:,} bars/stock")
    frames      = []
    sample_bars = None

    for i, tkr in enumerate(cfg["tickers"]):
        t0 = time.time()
        print(f"         [{i+1}/{len(cfg['tickers'])}] {tkr}... ", end="", flush=True)
        bars  = generate_bars(tkr, trading_days, cfg["bars_per_day"], seed=i*17)
        bsig  = compute_signals(bars, cfg)
        daily = aggregate_daily(bsig)
        daily["ticker"] = tkr
        daily["date"]   = daily.index
        frames.append(daily.reset_index(drop=True))
        if tkr == "AAPL":
            d0 = bsig["date"].iloc[0]
            sample_bars = bsig[bsig["date"] == d0].reset_index(drop=True)
        print(f"{len(daily)} days  ({time.time()-t0:.1f}s)")

    panel = pd.concat(frames, ignore_index=True).sort_values(["date","ticker"]).reset_index(drop=True)
    print(f"         Total: {len(panel):,} stock-days")

    # Step 2: Score
    print(f"\n  [2/5]  Building composite signal score")
    panel = build_score(panel)

    # Step 3: Positions
    print(f"\n  [3/5]  Position sizing (hold≥{cfg['holding_period_days']}d, "
          f"turn≤{cfg['max_daily_turnover']*100:.0f}%/day)")
    panel = build_positions(panel, cfg)
    pos_days = (panel["position"] > 0).sum()
    print(f"         Active long positions: {pos_days} stock-days")

    # Step 4: Backtest
    print(f"\n  [4/5]  Backtest (comm=10bps, impact=0bps)")
    res = run_backtest(panel, cfg)
    print(f"         Avg daily turnover: {res['turnover'].mean():.3%}")
    print(f"         Annual cost drag  : {res['cost'].mean()*252:.3%}")
    print(f"         NAV range         : {res['nav'].min():.4f} – {res['nav'].max():.4f}")

    # Step 5: Report
    print(f"\n  [5/5]  Analytics + API + Plot")
    met = metrics(res)
    sty = stylised_facts(res["net_pnl"])

    print("\n  ╔══════════════════════════════════════════════════════╗")
    print("  ║          PERFORMANCE SUMMARY  v3.0                  ║")
    print("  ╠════════════════════════════╦═════════════════════════╣")
    for k, v in {**met, **sty}.items():
        print(f"  ║  {k:<26}  ║  {v:>21}  ║")
    print("  ╚════════════════════════════╩═════════════════════════╝")

    print("\n  Calling Anthropic API...")
    ai_text = get_ai_commentary(met, sty, cfg)
    print(f"\n  ── Claude's Commentary ──\n{ai_text}")

    plot_all(res, panel, met, sty, cfg, sample_bars, ai_text)

    print(f"\n{'='*65}")
    print("  ✅  DONE — Microstructure_Backtest_Results.png saved")
    print(f"{'='*65}\n")
    return res, panel


if __name__ == "__main__":
    results, panel = main()
