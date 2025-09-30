"""
Bayesian/global optimization of a weighted-signal systematic strategy using bt for backtesting.

Key features:
- Data split: training vs test tickers. Training only used for optimization.
- Strategy: weighted sum of indicator-based boolean signals with buy/sell thresholds.
- Includes broad set of indicators via `ta` (RSI, Stoch, MACD, ROC, SMA/EMA cross, ADX, PSAR, BB, ATR, Donchian, OBV, CMF) + simple candlestick patterns.
- Trade execution: configurable TP/SL multipliers for long/short, executed on OHLC.
- Cross-validation: K-fold across training tickers.
- Objective: risk-adjusted composite (Sharpe/Sortino/Calmar) with MaxDD penalty and L1 regularization on weights.
- Optimizer: Optuna (Bayesian TPE). Fallback to skopt or random search if unavailable.
- Logging: CSV of params + metrics for both train/test for each trial.
- CLI: configure tickers, dates, intervals, optimization settings, CV splits.

Dependencies required (install via pip):
- bt
- yfinance
- ta
- optuna (optional, recommended)
- scikit-optimize (optional fallback)

Note: Designed to be reasonably efficient yet readable. You may tailor indicators and search space sizes to fit performance constraints.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import warnings
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend; no GUI windows
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# Third-party deps. We keep imports guarded to provide clearer error messages.
try:
    import bt  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("The 'bt' package is required. Install via: pip install bt") from e

try:
    import yfinance as yf  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("The 'yfinance' package is required. Install via: pip install yfinance") from e

try:
    import optuna  # type: ignore
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False

try:
    from skopt import gp_minimize  # type: ignore
    from skopt.space import Real, Integer  # type: ignore
    _HAS_SKOPT = True
except Exception:
    _HAS_SKOPT = False

try:
    import ta  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("The 'ta' package is required. Install via: pip install ta") from e

# Suppress known FutureWarning from ta.trend.PSARIndicator using Series.__setitem__
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Series.__setitem__ treating keys as positions is deprecated",
)


# -------------------------- Configuration dataclasses -------------------------

@dataclass
class OptimConfig:
    n_trials: int = 5000
    timeout: Optional[int] = None  # seconds
    cv_folds: int = 3
    l1_reg: float = 0.0  # L1 penalty on sum(abs(weights))
    max_dd_penalty: float = 0.25
    sharpe_weight: float = 1.0
    sortino_weight: float = 0.0
    calmar_weight: float = 0.5
    seed: int = 42


@dataclass
class DataConfig:
    tickers_train: List[str] = dataclasses.field(default_factory=lambda: ["AAPL", "MSFT", "GOOG", "AMZN", "FB", "BRK.B", "JNJ", "WMT", "V", "DIS", "PG", "INTC", "CSCO", "INTL", "AXP", "MRK"])
    tickers_test: List[str] = dataclasses.field(default_factory=lambda: ["META", "NVDA", "NFLX", "AMD", "TSLA", "QCOM"])
    start: str = "2015-01-01"
    end: str = "2025-01-01"
    interval: str = "1d"  # yfinance interval


@dataclass
class LoggingConfig:
    out_csv: str = "btbho_optim_log.csv"


# ------------------------------- Data Loading --------------------------------

def fetch_ohlcv(tickers: List[str], start: str, end: str, interval: str = "1d", split: str = "train") -> Dict[str, pd.DataFrame]:
    """Download OHLCV for each ticker. Returns dict[ticker] -> DataFrame[Open, High, Low, Close, Adj Close, Volume].
    Index is DatetimeIndex.
    """
    data: Dict[str, pd.DataFrame] = {}
    # Check for local split-level cache first
    cache_path = os.path.join(os.path.dirname(__file__), f"{split}_data")
    if os.path.exists(cache_path):
        try:
            cached = pd.read_pickle(cache_path)
            if isinstance(cached, dict) and all(isinstance(v, pd.DataFrame) for v in cached.values()):
                print(f"Using cached data for {split} split.")
                return cached  # early return using cached data
        except Exception:
            pass
    print(f"Downloading {len(tickers)} tickers for {split} split...")
    for t in tickers:
        df = yf.download(t, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            print(f"Failed to download data for {t}")
            continue
        # Normalize columns
        cols = {c: c for c in df.columns}
        if "Adj Close" not in df.columns and "Adj Close" not in cols:
            df["Adj Close"] = df["Close"]
        df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
        df.dropna(how="any", inplace=True)
        data[t] = df
    if not data:
        raise RuntimeError("No data downloaded. Check tickers or network.")
    # Save aggregated data to local split-level cache before returning
    try:
        pd.to_pickle(data, cache_path)
    except Exception:
        pass
    return data


# ------------------------------ Indicator Suite ------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Given a single-ticker OHLCV DataFrame, compute indicators as columns.
    Returns a DataFrame aligning with df.index.
    """
    # Ensure 1D Series even if columns are (n,1) DataFrames
    def _s(x):
        v = df[x]
        if isinstance(v, pd.DataFrame):
            return v.iloc[:, 0]
        return v

    close = _s("Close")
    high = _s("High")
    low = _s("Low")
    open_ = _s("Open")
    vol = _s("Volume")

    out = pd.DataFrame(index=df.index)

    # Momentum
    out["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()
    macd = ta.trend.MACD(close)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["roc_10"] = ta.momentum.ROCIndicator(close, window=10).roc()

    # Trend
    out["sma_50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    out["sma_200"] = ta.trend.SMAIndicator(close, window=200).sma_indicator()
    out["ema_20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    out["ema_50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    out["adx_14"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    psar = ta.trend.PSARIndicator(high, low, close)
    out["psar"] = psar.psar()

    # Volatility
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    out["bb_low"] = bb.bollinger_lband()
    out["bb_high"] = bb.bollinger_hband()
    out["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    don = ta.volatility.DonchianChannel(high, low, close, window=20)
    out["don_low"] = don.donchian_channel_lband()
    out["don_high"] = don.donchian_channel_hband()

    # Volume
    out["obv"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    out["cmf_20"] = ta.volume.ChaikinMoneyFlowIndicator(high, low, close, vol, window=20).chaikin_money_flow()

    # Candlestick patterns (simple heuristics)
    body = (close - open_).abs()
    rng = (high - low).replace(0, np.nan)
    upper_shadow = (high - close).where(close >= open_, (high - open_))
    lower_shadow = (open_ - low).where(close >= open_, (close - low))
    out["doji"] = (body <= 0.1 * rng).astype(int)
    out["hammer_bull"] = ((lower_shadow >= 2 * body) & (close > open_)).astype(int)
    out["hammer_bear"] = ((upper_shadow <= 2 * body) & (close < open_)).astype(int)
    out["hammer_bear"] = ((upper_shadow <= 2 * body) & (close < open_)).astype(int)
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    bullish_engulf = (close > open_) & (prev_close < prev_open) & (close >= prev_open) & (open_ <= prev_close)
    bearish_engulf = (close < open_) & (prev_close > prev_open) & (close <= prev_open) & (open_ >= prev_close)
    out["engulf_bull"] = bullish_engulf.astype(int)
    out["engulf_bear"] = bearish_engulf.astype(int)

    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return out


def build_signal_features(df: pd.DataFrame, ind: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Construct boolean signal features per the parameterized thresholds.
    Returns DataFrame with 0/1 features aligned with df index.
    """
    out = pd.DataFrame(index=df.index)

    # Threshold parameters (with defaults if missing)
    rsi_hi = params.get("rsi_hi", 70.0)
    rsi_lo = params.get("rsi_lo", 30.0)
    stoch_hi = params.get("stoch_hi", 80.0)
    stoch_lo = params.get("stoch_lo", 20.0)
    macd_hi = params.get("macd_hi", 0.0)
    macd_lo = params.get("macd_lo", 0.0)
    roc_hi = params.get("roc_hi", 0.0)
    roc_lo = params.get("roc_lo", 0.0)
    adx_hi = params.get("adx_hi", 25.0)
    cmf_hi = params.get("cmf_hi", 0.2)
    cmf_lo = params.get("cmf_lo", -0.2)

    # Helper: ensure df Close is a 1D Series and aligned to indicators' index
    _close = df["Close"]
    if isinstance(_close, pd.DataFrame):
        _close = _close.iloc[:, 0]
    _close = _close.reindex(ind.index).ffill()

    # Booleans
    out["rsi_gt_hi"] = (ind["rsi_14"] > rsi_hi).astype(int)
    out["rsi_lt_lo"] = (ind["rsi_14"] < rsi_lo).astype(int)
    out["stoch_gt_hi"] = (ind["stoch_k"] > stoch_hi).astype(int)
    out["stoch_lt_lo"] = (ind["stoch_k"] < stoch_lo).astype(int)
    out["macd_gt"] = (ind["macd"] > macd_hi).astype(int)
    out["macd_lt"] = (ind["macd"] < macd_lo).astype(int)
    out["roc_gt"] = (ind["roc_10"] > roc_hi).astype(int)
    out["roc_lt"] = (ind["roc_10"] < roc_lo).astype(int)

    # Crossovers
    sma_up = (ind["sma_50"] > ind["sma_200"]).astype(int)
    sma_dn = 1 - sma_up
    out["sma_cross_up"] = ((sma_up == 1) & (sma_up.shift(1) == 0)).astype(int)
    out["sma_cross_down"] = ((sma_dn == 1) & (sma_dn.shift(1) == 0)).astype(int)
    ema_up = (ind["ema_20"] > ind["ema_50"]).astype(int)
    ema_dn = 1 - ema_up
    out["ema_cross_up"] = ((ema_up == 1) & (ema_up.shift(1) == 0)).astype(int)
    out["ema_cross_down"] = ((ema_dn == 1) & (ema_dn.shift(1) == 0)).astype(int)

    # Trend/volatility/volume
    out["adx_gt"] = (ind["adx_14"] > adx_hi).astype(int)
    # Align Series before comparisons to avoid alignment errors
    psar = ind["psar"].reindex(ind.index)
    psar, c_al = psar.align(_close, join="left")
    out["psar_bull"] = (psar < c_al).astype(int)
    out["psar_bear"] = (psar > c_al).astype(int)
    bb_high = ind["bb_high"].reindex(ind.index)
    bb_low = ind["bb_low"].reindex(ind.index)
    _, c_high = bb_high.align(_close, join="left")
    _, c_low = bb_low.align(_close, join="left")
    out["close_gt_bb_high"] = (c_high > bb_high).astype(int)
    out["close_lt_bb_low"] = (c_low < bb_low).astype(int)
    don_high = ind["don_high"].reindex(ind.index)
    don_low = ind["don_low"].reindex(ind.index)
    _, c_dh = don_high.align(_close, join="left")
    _, c_dl = don_low.align(_close, join="left")
    out["breakout_up"] = (c_dh > don_high).astype(int)
    out["breakout_down"] = (c_dl < don_low).astype(int)

    obv_diff = ind["obv"].diff().fillna(0)
    out["obv_slope_pos"] = (obv_diff > 0).astype(int)
    out["obv_slope_neg"] = (obv_diff < 0).astype(int)
    out["cmf_gt"] = (ind["cmf_20"] > cmf_hi).astype(int)
    out["cmf_lt"] = (ind["cmf_20"] < cmf_lo).astype(int)

    # Candlestick
    out["doji"] = ind["doji"].astype(int)
    out["hammer_bull"] = ind["hammer_bull"].astype(int)
    out["hammer_bear"] = ind["hammer_bear"].astype(int)
    out["engulf_bull"] = ind["engulf_bull"].astype(int)
    out["engulf_bear"] = ind["engulf_bear"].astype(int)

    return out.fillna(0).astype(int)


def compute_weighted_score(features: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Compute weighted sum score per date given 0/1 features and weights dict."""
    missing_cols = [c for c in weights.keys() if c not in features.columns]
    if missing_cols:
        raise ValueError(f"Missing features for weights: {missing_cols}")
    mat = features[sorted(weights.keys())].astype(float)
    w = np.array([weights[c] for c in sorted(weights.keys())], dtype=float)
    score = pd.Series(mat.values @ w, index=features.index)
    return score


# --------------------------- Metrics and Objective ---------------------------

def equity_from_positions(df: pd.DataFrame, position: pd.Series, start_equity: float = 1.0) -> pd.Series:
    """Compute equity curve given positions and close-to-close returns."""
    ret = df["Close"].pct_change().fillna(0.0)
    pnl = ret * position.shift(1).fillna(0.0)
    equity = (1.0 + pnl).cumprod() * start_equity
    return equity


def metrics_from_equity(equity: pd.Series) -> Dict[str, float]:
    ret = equity.pct_change().dropna()
    if ret.empty:
        return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "max_dd": 1.0, "cagr": 0.0}
    mean = ret.mean()
    std = ret.std(ddof=0)
    downside = ret[ret < 0]
    dd_std = downside.std(ddof=0) if not downside.empty else 0.0
    sharpe = (mean / std) * math.sqrt(252) if std > 1e-12 else 0.0
    sortino = (mean / dd_std) * math.sqrt(252) if dd_std > 1e-12 else 0.0
    roll_max = equity.cummax()
    drawdown = (equity / roll_max) - 1.0
    max_dd = drawdown.min() if not drawdown.empty else 0.0
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-6)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    return {"sharpe": sharpe, "sortino": sortino, "calmar": calmar, "max_dd": float(abs(max_dd)), "cagr": cagr}


def composite_objective(metrics: Dict[str, float], cfg: OptimConfig, l1_penalty: float) -> float:
    score = (
        cfg.sharpe_weight * metrics.get("sharpe", 0.0)
        + cfg.sortino_weight * metrics.get("sortino", 0.0)
        + cfg.calmar_weight * metrics.get("calmar", 0.0)
    )
    score -= cfg.max_dd_penalty * metrics.get("max_dd", 0.0)
    score -= l1_penalty
    return float(score)


# --------------------------- Strategy construction ---------------------------

FEATURE_LIST = [
    # Momentum
    "rsi_gt_hi", "rsi_lt_lo", "stoch_gt_hi", "stoch_lt_lo", "macd_gt", "macd_lt", "roc_gt", "roc_lt",
    # Trend
    "sma_cross_up", "sma_cross_down", "ema_cross_up", "ema_cross_down", "adx_gt", "psar_bull", "psar_bear",
    # Volatility
    "close_gt_bb_high", "close_lt_bb_low", "breakout_up", "breakout_down",
    # Volume
    "obv_slope_pos", "obv_slope_neg", "cmf_gt", "cmf_lt",
    # Candles
    "doji", "hammer_bull", "hammer_bear", "engulf_bull", "engulf_bear",
]


def build_weights_from_params(params: dict) -> Dict[str, float]:
    weights = {}
    for feat in FEATURE_LIST:
        weights[feat] = float(params.get(f"w_{feat}", 0.0))
    return weights


def positions_for_ticker(df: pd.DataFrame, params: dict) -> pd.Series:
    ind = compute_indicators(df)
    feats = build_signal_features(df, ind, params)
    weights = build_weights_from_params(params)
    score = compute_weighted_score(feats, weights)
    # Position is set to be the score rounded to the closest 5 capped between -10 and 10
    pos = np.round(score / 5) * 5
    pos = pos.clip(lower=-10, upper=10)
    return pos


def build_weights_matrix(data: Dict[str, pd.DataFrame], params: dict) -> pd.DataFrame:
    positions = {}
    for t, df in data.items():
        positions[t] = positions_for_ticker(df, params)
    if not positions:
        raise RuntimeError("No positions generated for any ticker.")
    weights = pd.DataFrame(positions).fillna(0.0)
    return weights


def _align_close_matrix(weights: pd.DataFrame, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align and assemble Close price matrix for tickers in `weights`.

    Returns DataFrame indexed as `weights.index`, columns as `weights.columns` (subset that exists in data),
    forward-filled to handle missing values.
    """
    series_list = []
    for t in weights.columns:
        if t not in data:
            continue
        s = data[t]["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.reindex(weights.index).ffill()
        s.name = t
        series_list.append(s)
    if not series_list:
        raise ValueError("No overlapping tickers between weights and data to build price matrix.")
    price = pd.concat(series_list, axis=1).reindex(weights.index)
    return price


def _simulate_ticker_from_targets(close: pd.Series, targets: pd.Series) -> pd.DataFrame:
    """Simulate per-ticker cashflows and realized PnL from daily target units using close prices.

    targets: desired signed units each day in the set {-10, -5, 0, 5, 10} (or similar ints).
    Mechanics:
    - If desired increases on the same side, add units at today's close and update average entry price.
    - If desired decreases on the same side, realize PnL for the reduced units at today's close.
    - If sign flips, fully close old side at today's close (realize), then open new side at today's close.
    - If desired is unchanged, do nothing (no realized PnL).

    Realized PnL formula (unit-returns):
    - Long close of q units: q * (exit/avg_entry - 1)
    - Short close of q units: q * (avg_entry/exit - 1)

    Returns a DataFrame per day with columns:
      - realized_pnl: cash PnL realized that day
      - spent: cash outflow that day (e.g., buy to open/increase long, buy to close short)
      - received: cash inflow that day (e.g., sell to reduce/close long, sell to open/increase short)
    """
    close = close.astype(float)
    targets = targets.astype(float).fillna(0.0)
    idx = close.index
    n = len(idx)
    realized = np.zeros(n, dtype=float)
    spent = np.zeros(n, dtype=float)
    received = np.zeros(n, dtype=float)

    side = 0  # 0 flat, +1 long, -1 short
    units = 0.0
    avg_entry = 0.0

    for i, dt in enumerate(idx):
        px = float(close.iloc[i])
        desired = float(targets.iloc[i])

        current_signed = side * units
        delta = desired - current_signed

        if abs(delta) < 1e-12:
            # No change
            continue

        # Helpers for cashflows
        def realize_pnl(q: float, s: int, avg: float, price: float) -> float:
            """Cash PnL for closing q units of side s at price."""
            if q <= 0 or s == 0:
                return 0.0
            if s > 0:  # long
                return q * (price - avg)
            else:  # short
                return q * (avg - price)

        if side == 0:
            # Opening new position from flat
            if abs(desired) > 0:
                side = 1 if desired > 0 else -1
                units = abs(desired)
                avg_entry = px
                if side > 0:
                    # Buy to open
                    spent[i] += units * px
                else:
                    # Sell to open (short)
                    received[i] += units * px
        else:
            desired_side = 0 if abs(desired) == 0 else (1 if desired > 0 else -1)
            if desired_side == side or desired_side == 0:
                # Same side or reducing to zero without flipping
                desired_units = abs(desired)
                if desired_units > units + 1e-12:
                    # Add units at px, update average entry
                    add = desired_units - units
                    avg_entry = (avg_entry * units + px * add) / (units + add)
                    units = desired_units
                    if side > 0:
                        # Buy more long
                        spent[i] += add * px
                    else:
                        # Increase short -> sell more
                        received[i] += add * px
                elif desired_units < units - 1e-12:
                    # Close partial units at px
                    close_q = units - desired_units
                    realized[i] += realize_pnl(close_q, side, avg_entry, px)
                    if side > 0:
                        # Sell to reduce long -> receive cash
                        received[i] += close_q * px
                    else:
                        # Buy to reduce short -> spend cash
                        spent[i] += close_q * px
                    units = desired_units
                    if units == 0:
                        side = 0
                        avg_entry = 0.0
                else:
                    # No change
                    pass
            else:
                # Sign flip: fully close, then open new side
                if units > 0:
                    realized[i] += realize_pnl(units, side, avg_entry, px)
                    if side > 0:
                        # Close long -> sell
                        received[i] += units * px
                    else:
                        # Close short -> buy
                        spent[i] += units * px
                # Open new side with remaining desired units
                side = desired_side
                units = abs(desired)
                avg_entry = px if units > 0 else 0.0
                if units > 0:
                    if side > 0:
                        spent[i] += units * px  # buy to open long
                    else:
                        received[i] += units * px  # sell to open short

    return pd.DataFrame({
        "realized_pnl": realized,
        "spent": spent,
        "received": received,
    }, index=idx)


def run_bt_backtest(weights: pd.DataFrame, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Custom backtest using daily target units from `weights` and close prices.

    For each ticker, simulate inventory transitions to match the target units per day and compute
    realized PnL at the close. Aggregate realized PnL across tickers to a portfolio daily PnL, then
    build an equity curve (starting at 1.0) as 1.0 + cumulative realized PnL.

    Note: Units are abstract; metrics scale with the chosen unit magnitudes. Use consistent targets
    (e.g., -10,-5,0,5,10) across tickers.
    """
    if weights is None or weights.empty:
        raise ValueError("weights is empty")

    prices = _align_close_matrix(weights, data)
    weights = weights.reindex(prices.index).fillna(0.0)

    # Compute per-ticker cashflows
    pnl_parts = []
    for t in weights.columns:
        if t not in prices.columns:
            continue
        cf = _simulate_ticker_from_targets(prices[t], weights[t])
        cf.columns = pd.MultiIndex.from_product([[t], cf.columns])
        pnl_parts.append(cf)

    if not pnl_parts:
        raise ValueError("No realized PnL components computed.")

    per_ticker = pd.concat(pnl_parts, axis=1)
    # Aggregate across tickers
    realized = per_ticker.xs("realized_pnl", axis=1, level=1).sum(axis=1)
    spent = per_ticker.xs("spent", axis=1, level=1).sum(axis=1)
    received = per_ticker.xs("received", axis=1, level=1).sum(axis=1)
    cum_pnl = realized.cumsum()
    equity = 1.0 + cum_pnl
    out = pd.DataFrame({
        "equity": equity,
        "cum_pnl": cum_pnl,
        "realized_pnl": realized,
        "spent": spent,
        "received": received,
    }, index=equity.index)
    return out


def evaluate_params(
    params: dict,
    data_train: Dict[str, pd.DataFrame],
    data_test: Dict[str, pd.DataFrame],
    cfg: OptimConfig,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Return objective score, train metrics, test metrics."""
    # CV across training tickers
    tickers = list(data_train.keys())
    random.Random(cfg.seed).shuffle(tickers)
    k = max(1, min(cfg.cv_folds, len(tickers)))
    folds = [tickers[i::k] for i in range(k)]

    train_scores = []
    agg_train_metrics = {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "max_dd": 0.0, "cagr": 0.0}
    for fold in folds:
        data_fold = {t: data_train[t] for t in fold}
        weights = build_weights_matrix(data_fold, params)
        # Custom simulator returns DataFrame with equity, cum_pnl, realized_pnl, spent, received
        bt_df = run_bt_backtest(weights, data_fold)
        m = metrics_from_equity(bt_df["equity"])  # compute metrics on equity

        # Print cashflow stats
        try:
            max_spent = float(bt_df["spent"].max())
            max_received = float(bt_df["received"].max())
            max_pnl_day = float(bt_df["realized_pnl"].max())
            min_pnl_day = float(bt_df["realized_pnl"].min())
            print(f"Fold {fold}: max spent/day={max_spent:.2f}, max received/day={max_received:.2f}, max pnl/day={max_pnl_day:.2f}, min pnl/day={min_pnl_day:.2f}")
        except Exception:
            pass

        # Ensure finite metrics and objective
        m = {k: float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for k, v in m.items()}
        for k2, v in m.items():
            agg_train_metrics[k2] += v
        obj = composite_objective(m, cfg, l1_penalty=l1_from_params(params, cfg.l1_reg))
        if not np.isfinite(obj):
            obj = -1e9
        train_scores.append(obj)
    # Average metrics across folds
    for k2 in agg_train_metrics:
        agg_train_metrics[k2] /= max(len(folds), 1)
    train_obj = float(np.mean(train_scores)) if train_scores else -1e9

    # Test evaluation (no influence on objective)
    test_metrics = {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "max_dd": 0.0, "cagr": 0.0}
    if data_test:
        try:
            weights_test = build_weights_matrix(data_test, params)
            bt_t_df = run_bt_backtest(weights_test, data_test)
            test_metrics = metrics_from_equity(bt_t_df["equity"])  # metrics from equity
            # Cashflow summary on test
            test_metrics["final_cum_pnl"] = float(bt_t_df["cum_pnl"].iloc[-1]) if not bt_t_df.empty else 0.0
            test_metrics["max_spent_day"] = float(bt_t_df["spent"].max()) if not bt_t_df.empty else 0.0
            test_metrics["ave_spent_day"] = float(bt_t_df["spent"].mean()) if not bt_t_df.empty else 0.0
            test_metrics["total_spent"] = float(bt_t_df["spent"].sum()) if not bt_t_df.empty else 0.0
            test_metrics["max_received_day"] = float(bt_t_df["received"].max()) if not bt_t_df.empty else 0.0
            test_metrics["ave_received_day"] = float(bt_t_df["received"].mean()) if not bt_t_df.empty else 0.0
            test_metrics["total_received"] = float(bt_t_df["received"].sum()) if not bt_t_df.empty else 0.0
            test_metrics["ave_pnl_day"] = float(bt_t_df["realized_pnl"].mean()) if not bt_t_df.empty else 0.0
            test_metrics["total_pnl"] = float(bt_t_df["realized_pnl"].sum()) if not bt_t_df.empty else 0.0
            test_metrics["max_pnl_day"] = float(bt_t_df["realized_pnl"].max()) if not bt_t_df.empty else 0.0
            test_metrics["min_pnl_day"] = float(bt_t_df["realized_pnl"].min()) if not bt_t_df.empty else 0.0
        except Exception:
            pass

    return train_obj, agg_train_metrics, test_metrics


def l1_from_params(params: dict, l1_lambda: float) -> float:
    if l1_lambda <= 0:
        return 0.0
    w = build_weights_from_params(params)
    return l1_lambda * float(np.sum(np.abs(np.fromiter(w.values(), dtype=float))))


# ------------------------------- Optimization --------------------------------

def suggest_params_optuna(trial: "optuna.Trial") -> dict:
    params = {}
    # Thresholds
    params["rsi_hi"] = trial.suggest_float("rsi_hi", 60.0, 90.0)
    params["rsi_lo"] = trial.suggest_float("rsi_lo", 10.0, 40.0)
    params["stoch_hi"] = trial.suggest_float("stoch_hi", 70.0, 95.0)
    params["stoch_lo"] = trial.suggest_float("stoch_lo", 5.0, 35.0)
    params["macd_hi"] = trial.suggest_float("macd_hi", -1.0, 1.0)
    params["macd_lo"] = trial.suggest_float("macd_lo", -1.0, 1.0)
    params["roc_hi"] = trial.suggest_float("roc_hi", -1.0, 1.0)
    params["roc_lo"] = trial.suggest_float("roc_lo", -1.0, 1.0)
    params["adx_hi"] = trial.suggest_float("adx_hi", 15.0, 40.0)
    params["cmf_hi"] = trial.suggest_float("cmf_hi", 0.0, 0.5)
    params["cmf_lo"] = trial.suggest_float("cmf_lo", -0.5, 0.0)

    # Decision thresholds
    params["buy_threshold"] = trial.suggest_float("buy_threshold", 1.0, 6.0)
    params["sell_threshold"] = trial.suggest_float("sell_threshold", -6.0, -1.0)

    # Feature weights
    for feat in FEATURE_LIST:
        params[f"w_{feat}"] = trial.suggest_float(f"w_{feat}", -2.0, 2.0)

    return params


def random_param_sample(rng: random.Random) -> dict:
    def u(a, b):
        return rng.random() * (b - a) + a
    params = {
        "rsi_hi": u(60, 90),
        "rsi_lo": u(10, 40),
        "stoch_hi": u(70, 95),
        "stoch_lo": u(5, 35),
        "macd_hi": u(-1, 1),
        "macd_lo": u(-1, 1),
        "roc_hi": u(-1, 1),
        "roc_lo": u(-1, 1),
        "adx_hi": u(15, 40),
        "cmf_hi": u(0, 0.5),
        "cmf_lo": u(-0.5, 0),
        "buy_threshold": u(1.0, 6.0),
        "sell_threshold": u(-6.0, -1.0),
    }
    for feat in FEATURE_LIST:
        params[f"w_{feat}"] = u(-2.0, 2.0)
    return params


def optimize_params(
    data_train: Dict[str, pd.DataFrame],
    data_test: Dict[str, pd.DataFrame],
    cfg: OptimConfig,
    log_cfg: LoggingConfig,
) -> Tuple[dict, float]:
    rng = random.Random(cfg.seed)

    os.makedirs(os.path.dirname(log_cfg.out_csv) or ".", exist_ok=True)
    if not os.path.exists(log_cfg.out_csv):
        pd.DataFrame().to_csv(log_cfg.out_csv, index=False)

    def evaluate_and_log(params: dict, trial_number: int) -> float:
        train_obj, train_metrics, test_metrics = evaluate_params(params, data_train, data_test, cfg)
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "trial": trial_number,
            "train_obj": train_obj,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
            **params,
        }
        try:
            pd.DataFrame([row]).to_csv(log_cfg.out_csv, mode="a", header=not os.path.getsize(log_cfg.out_csv), index=False)
        except Exception:
            pass
        return -train_obj  # minimization in some optimizers; optuna will maximize explicitly below

    best_params: dict = {}
    best_score: float = -1e18

    if _HAS_OPTUNA:
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: "optuna.Trial") -> float:
            params = suggest_params_optuna(trial)
            train_obj, train_metrics, test_metrics = evaluate_params(params, data_train, data_test, cfg)
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "trial": trial.number,
                "train_obj": train_obj,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **params,
            }
            try:
                pd.DataFrame([row]).to_csv(log_cfg.out_csv, mode="a", header=not os.path.getsize(log_cfg.out_csv), index=False)
            except Exception:
                pass
            return train_obj

        study.optimize(objective, n_trials=cfg.n_trials, timeout=cfg.timeout)
        if study.best_trial is not None:
            best_params = study.best_trial.params
            best_score = study.best_value
    elif _HAS_SKOPT:
        # Build search space for skopt
        space = [
            Real(60, 90, name="rsi_hi"), Real(10, 40, name="rsi_lo"),
            Real(70, 95, name="stoch_hi"), Real(5, 35, name="stoch_lo"),
            Real(-1, 1, name="macd_hi"), Real(-1, 1, name="macd_lo"),
            Real(-1, 1, name="roc_hi"), Real(-1, 1, name="roc_lo"),
            Real(15, 40, name="adx_hi"), Real(0, 0.5, name="cmf_hi"), Real(-0.5, 0, name="cmf_lo"),
            Real(1.0, 6.0, name="buy_threshold"), Real(-6.0, -1.0, name="sell_threshold"),
        ] + [Real(-2.0, 2.0, name=f"w_{f}") for f in FEATURE_LIST]

        def skopt_objective(x):
            keys = [s.name for s in space]
            params = {k: float(v) for k, v in zip(keys, x)}
            return evaluate_and_log(params, trial_number=-1)

        res = gp_minimize(skopt_objective, space, n_calls=cfg.n_trials, random_state=cfg.seed)
        keys = [s.name for s in space]
        best_params = {k: float(v) for k, v in zip(keys, res.x)}
        best_score = -res.fun
    else:
        # Random search fallback
        for i in range(cfg.n_trials):
            params = random_param_sample(rng)
            score = -evaluate_and_log(params, trial_number=i)
            if score > best_score:
                best_score = score
                best_params = params

    return best_params, best_score


# ----------------------------------- CLI -------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BHO for weighted-signal strategy using bt")
    p.add_argument("--train", type=str, default="AAPL,MSFT,GOOG,AMZN", help="Comma-separated training tickers")
    p.add_argument("--test", type=str, default="META,NVDA", help="Comma-separated test tickers")
    p.add_argument("--start", type=str, default="2015-01-01")
    p.add_argument("--end", type=str, default="2025-01-01")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--trials", type=int, default=5Buy 50000)
    p.add_argument("--cv", type=int, default=3, help="CV folds across training tickers")
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--l1", type=float, default=0.0, help="L1 regularization lambda")
    p.add_argument("--dd_pen", type=float, default=0.25, help="Max drawdown penalty weight")
    p.add_argument("--sharpe_w", type=float, default=1.0)
    p.add_argument("--sortino_w", type=float, default=0.0)
    p.add_argument("--calmar_w", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_csv", type=str, default="btbho_optim_log.csv")
    return p.parse_args()


def main():
    args = parse_args()

    data_cfg = DataConfig(
        tickers_train=[t.strip().upper() for t in args.train.split(",") if t.strip()],
        tickers_test=[t.strip().upper() for t in args.test.split(",") if t.strip()],
        start=args.start,
        end=args.end,
        interval=args.interval,
    )
    opt_cfg = OptimConfig(
        n_trials=args.trials,
        timeout=args.timeout,
        cv_folds=args.cv,
        l1_reg=args.l1,
        max_dd_penalty=args.dd_pen,
        sharpe_weight=args.sharpe_w,
        sortino_weight=args.sortino_w,
        calmar_weight=args.calmar_w,
        seed=args.seed,
    )
    log_cfg = LoggingConfig(out_csv=args.log_csv)

    print("Downloading data...")
    data_train = fetch_ohlcv(data_cfg.tickers_train, data_cfg.start, data_cfg.end, data_cfg.interval, split="train")
    data_test = fetch_ohlcv(data_cfg.tickers_test, data_cfg.start, data_cfg.end, data_cfg.interval, split="test")

    print("Starting optimization...")
    best_params, best_score = optimize_params(data_train, data_test, opt_cfg, log_cfg)
    print("Best score:", best_score)
    print("Best params:")
    print(json.dumps(best_params, indent=2))

    # Final evaluation report
    train_obj, train_metrics, test_metrics = evaluate_params(best_params, data_train, data_test, opt_cfg)
    print("Final Train metrics:", train_metrics)
    print("Final Test metrics:", test_metrics)


if __name__ == "__main__":
    main()

