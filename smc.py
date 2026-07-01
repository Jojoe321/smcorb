"""
smc.py - Smart Money Concepts (SMC) Market Structure Analysis.

Implements institutional price-action analysis tools:
  1. Swing High/Low detection (fractal-based)
  2. Break of Structure (BOS) detection
  3. Change of Character (CHoCH) detection
  4. Order Block (OB) identification
  5. Fair Value Gap (FVG) detection
  6. Liquidity Sweep detection

All functions accept a pandas DataFrame with OHLC columns and return
structured data (dicts/lists) suitable for downstream combination
with ORB signals.

DESIGN NOTES:
  - "Lookback" parameters control sensitivity vs. noise. Higher values
    produce fewer, more significant swing points.
  - BOS vs. CHoCH distinction: BOS continues the trend, CHoCH reverses it.
  - Order blocks are identified retroactively after a BOS occurs.
  - FVGs use the standard 3-candle imbalance pattern.
  - All functions are stateless - they operate on the full DataFrame
    each call. For incremental/streaming use, slice the buffer.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from utils import setup_logging

logger = setup_logging()


# ──────────────────────────────────────────────────────────────
# 1. SWING HIGH / LOW DETECTION
# ──────────────────────────────────────────────────────────────

def detect_swing_points(
    df: pd.DataFrame,
    lookback: int = 5
) -> pd.DataFrame:
    """
    Identify swing highs and swing lows using a fractal-based approach.

    A swing high occurs when a bar's high is the HIGHEST high among
    the surrounding `lookback` bars on each side. Similarly, a swing
    low occurs when a bar's low is the LOWEST low on each side.

    This is the foundation for all structural analysis - BOS, CHoCH,
    order blocks, and liquidity sweeps all depend on accurate swing
    point identification.

    Args:
        df: DataFrame with columns ['high', 'low'] and a DatetimeIndex.
        lookback: Number of bars to check on each side of the candidate.
                  Higher values = fewer, more significant swings.
                  Typical: 3 (noisy) to 10 (major structure only).

    Returns:
        A copy of the DataFrame with additional columns:
          - 'swing_high' (bool): True at swing high bars
          - 'swing_low' (bool): True at swing low bars
          - 'swing_high_price' (float): The high price at swing highs (NaN elsewhere)
          - 'swing_low_price' (float): The low price at swing lows (NaN elsewhere)
    """
    result = df.copy()
    n = len(result)

    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    highs = result["high"].values
    lows = result["low"].values

    for i in range(lookback, n - lookback):
        # Swing High: current high is strictly greater than all
        # highs in the lookback window on both sides
        left_highs = highs[i - lookback:i]
        right_highs = highs[i + 1:i + lookback + 1]

        if highs[i] > left_highs.max() and highs[i] > right_highs.max():
            swing_high[i] = True

        # Swing Low: current low is strictly less than all lows
        # in the lookback window on both sides
        left_lows = lows[i - lookback:i]
        right_lows = lows[i + 1:i + lookback + 1]

        if lows[i] < left_lows.min() and lows[i] < right_lows.min():
            swing_low[i] = True

    result["swing_high"] = swing_high
    result["swing_low"] = swing_low
    result["swing_high_price"] = np.where(swing_high, highs, np.nan)
    result["swing_low_price"] = np.where(swing_low, lows, np.nan)

    sh_count = swing_high.sum()
    sl_count = swing_low.sum()
    logger.debug(
        f"Swing detection (lookback={lookback}): "
        f"{sh_count} highs, {sl_count} lows in {n} bars"
    )

    return result


def get_swing_list(df_with_swings: pd.DataFrame) -> list[dict]:
    """
    Extract a chronological list of swing points from the annotated
    DataFrame (output of detect_swing_points).

    Returns:
        List of dicts: [{type: 'high'|'low', price: float, time: datetime}, ...]
        Sorted by time.
    """
    swings = []

    sh_mask = df_with_swings["swing_high"] == True
    for idx, row in df_with_swings.loc[sh_mask].iterrows():
        swings.append({
            "type": "high",
            "price": float(row["high"]),
            "time": idx,
        })

    sl_mask = df_with_swings["swing_low"] == True
    for idx, row in df_with_swings.loc[sl_mask].iterrows():
        swings.append({
            "type": "low",
            "price": float(row["low"]),
            "time": idx,
        })

    swings.sort(key=lambda s: s["time"])
    return swings


# ──────────────────────────────────────────────────────────────
# 2. BREAK OF STRUCTURE (BOS)
# ──────────────────────────────────────────────────────────────

def detect_bos(
    df: pd.DataFrame,
    swings: list[dict]
) -> list[dict]:
    """
    Detect Break of Structure (BOS) events.

    BOS occurs when price continues the prevailing trend by breaking
    past a swing point in the trend direction:
      - Bullish BOS: price closes above the most recent swing HIGH
        (continuation of uptrend - making a higher high)
      - Bearish BOS: price closes below the most recent swing LOW
        (continuation of downtrend - making a lower low)

    The key distinction from CHoCH: BOS confirms the existing trend,
    it does NOT reverse it.

    Args:
        df: DataFrame with OHLC and DatetimeIndex.
        swings: Sorted list of swing points from get_swing_list().

    Returns:
        List of BOS events:
        [{type: 'BOS', direction: 'bullish'|'bearish',
          break_price: float, break_time: datetime,
          swing_ref: dict (the swing that was broken)}, ...]
    """
    if len(swings) < 4:
        return []

    bos_events = []

    for i in range(1, len(swings)):
        current = swings[i]

        if current["type"] == "high":
            # Find previous swing high
            prev_same = None
            for j in range(i - 1, -1, -1):
                if swings[j]["type"] == "high":
                    prev_same = swings[j]
                    break

            if prev_same is None:
                continue

            # Bullish BOS: current swing high > previous swing high
            # (Higher High - trend continuation)
            if current["price"] > prev_same["price"]:
                break_level = prev_same["price"]
                bars_after = df.loc[df.index > prev_same["time"]]

                for idx, bar in bars_after.iterrows():
                    if idx > current["time"]:
                        break
                    if bar["close"] > break_level:
                        bos_events.append({
                            "type": "BOS",
                            "direction": "bullish",
                            "break_price": float(bar["close"]),
                            "break_time": idx,
                            "swing_ref": prev_same,
                            "level": break_level,
                        })
                        break

        else:  # current is a swing low
            prev_same = None
            for j in range(i - 1, -1, -1):
                if swings[j]["type"] == "low":
                    prev_same = swings[j]
                    break

            if prev_same is None:
                continue

            # Bearish BOS: current swing low < previous swing low
            # (Lower Low - trend continuation)
            if current["price"] < prev_same["price"]:
                break_level = prev_same["price"]
                bars_after = df.loc[df.index > prev_same["time"]]

                for idx, bar in bars_after.iterrows():
                    if idx > current["time"]:
                        break
                    if bar["close"] < break_level:
                        bos_events.append({
                            "type": "BOS",
                            "direction": "bearish",
                            "break_price": float(bar["close"]),
                            "break_time": idx,
                            "swing_ref": prev_same,
                            "level": break_level,
                        })
                        break

    logger.debug(f"BOS detection: found {len(bos_events)} events")
    return bos_events


# ──────────────────────────────────────────────────────────────
# 3. CHANGE OF CHARACTER (CHoCH)
# ──────────────────────────────────────────────────────────────

def detect_choch(
    df: pd.DataFrame,
    swings: list[dict]
) -> list[dict]:
    """
    Detect Change of Character (CHoCH) events.

    CHoCH signals a potential trend REVERSAL. It occurs when price
    breaks structure AGAINST the prevailing trend:
      - Bearish CHoCH: in an uptrend (HH/HL sequence), price breaks
        below the most recent swing LOW (Higher Low broken)
      - Bullish CHoCH: in a downtrend (LH/LL sequence), price breaks
        above the most recent swing HIGH (Lower High broken)

    This is the opposite of BOS - instead of continuing the trend,
    it invalidates it.

    Args:
        df: DataFrame with OHLC and DatetimeIndex.
        swings: Sorted list of swing points from get_swing_list().

    Returns:
        List of CHoCH events:
        [{type: 'CHoCH', direction: 'bullish'|'bearish',
          break_price: float, break_time: datetime,
          swing_ref: dict, prior_trend: 'bullish'|'bearish'}, ...]
    """
    if len(swings) < 4:
        return []

    choch_events = []

    for i in range(3, len(swings)):
        recent = swings[max(0, i - 5):i + 1]

        recent_highs = [s for s in recent if s["type"] == "high"]
        recent_lows = [s for s in recent if s["type"] == "low"]

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            continue

        last_two_highs = recent_highs[-2:]
        last_two_lows = recent_lows[-2:]

        highs_rising = last_two_highs[1]["price"] > last_two_highs[0]["price"]
        lows_rising = last_two_lows[1]["price"] > last_two_lows[0]["price"]

        current = swings[i]

        if current["type"] == "low":
            # Check for bearish CHoCH: uptrend broken
            if highs_rising or lows_rising:
                last_swing_low = last_two_lows[-2]

                if current["price"] < last_swing_low["price"]:
                    break_level = last_swing_low["price"]
                    bars_after = df.loc[df.index > last_swing_low["time"]]

                    for idx, bar in bars_after.iterrows():
                        if idx > current["time"]:
                            break
                        if bar["close"] < break_level:
                            choch_events.append({
                                "type": "CHoCH",
                                "direction": "bearish",
                                "break_price": float(bar["close"]),
                                "break_time": idx,
                                "swing_ref": last_swing_low,
                                "prior_trend": "bullish",
                                "level": break_level,
                            })
                            break

        elif current["type"] == "high":
            # Check for bullish CHoCH: downtrend broken
            highs_falling = last_two_highs[1]["price"] < last_two_highs[0]["price"]
            lows_falling = last_two_lows[1]["price"] < last_two_lows[0]["price"]

            if highs_falling or lows_falling:
                last_swing_high = last_two_highs[-2]

                if current["price"] > last_swing_high["price"]:
                    break_level = last_swing_high["price"]
                    bars_after = df.loc[df.index > last_swing_high["time"]]

                    for idx, bar in bars_after.iterrows():
                        if idx > current["time"]:
                            break
                        if bar["close"] > break_level:
                            choch_events.append({
                                "type": "CHoCH",
                                "direction": "bullish",
                                "break_price": float(bar["close"]),
                                "break_time": idx,
                                "swing_ref": last_swing_high,
                                "prior_trend": "bearish",
                                "level": break_level,
                            })
                            break

    logger.debug(f"CHoCH detection: found {len(choch_events)} events")
    return choch_events



# ──────────────────────────────────────────────────────────────
# 4. ORDER BLOCK (OB) IDENTIFICATION
# ──────────────────────────────────────────────────────────────

def detect_order_blocks(
    df: pd.DataFrame,
    bos_events: list[dict],
    ob_lookback: int = 10
) -> list[dict]:
    """
    Identify Order Blocks (OBs) based on BOS events.

    An order block is the LAST OPPOSING CANDLE before an impulsive
    move that caused a Break of Structure. The theory is that
    institutional orders were placed at that candle, creating a
    supply/demand zone.

    For a BULLISH BOS:
      - Look backwards for the last BEARISH candle (close < open).
        That candle's body range = bullish OB (demand zone).

    For a BEARISH BOS:
      - Look backwards for the last BULLISH candle (close > open).
        That candle's body range = bearish OB (supply zone).

    Mitigation tracking:
      - An OB is "mitigated" once price returns into the zone.

    Args:
        df: DataFrame with OHLC and DatetimeIndex.
        bos_events: List of BOS events from detect_bos().
        ob_lookback: Max bars to look back from BOS for the OB candle.

    Returns:
        List of order block dicts.
    """
    order_blocks = []

    for bos in bos_events:
        bos_time = bos["break_time"]
        bars_before = df.loc[df.index < bos_time].tail(ob_lookback)

        if bars_before.empty:
            continue

        ob_candle = None

        if bos["direction"] == "bullish":
            for idx in reversed(bars_before.index):
                bar = bars_before.loc[idx]
                if bar["close"] < bar["open"]:
                    ob_candle = {
                        "type": "OB",
                        "direction": "bullish",
                        "top": float(max(bar["open"], bar["close"])),
                        "bottom": float(min(bar["open"], bar["close"])),
                        "ob_high": float(bar["high"]),
                        "ob_low": float(bar["low"]),
                        "ob_time": idx,
                        "bos_time": bos_time,
                    }
                    break

        elif bos["direction"] == "bearish":
            for idx in reversed(bars_before.index):
                bar = bars_before.loc[idx]
                if bar["close"] > bar["open"]:
                    ob_candle = {
                        "type": "OB",
                        "direction": "bearish",
                        "top": float(max(bar["open"], bar["close"])),
                        "bottom": float(min(bar["open"], bar["close"])),
                        "ob_high": float(bar["high"]),
                        "ob_low": float(bar["low"]),
                        "ob_time": idx,
                        "bos_time": bos_time,
                    }
                    break

        if ob_candle is not None:
            bars_after = df.loc[df.index > bos_time]
            ob_candle["mitigated"] = False
            ob_candle["mitigation_time"] = None

            for idx, bar in bars_after.iterrows():
                if ob_candle["direction"] == "bullish":
                    if bar["low"] <= ob_candle["top"]:
                        ob_candle["mitigated"] = True
                        ob_candle["mitigation_time"] = idx
                        break
                else:
                    if bar["high"] >= ob_candle["bottom"]:
                        ob_candle["mitigated"] = True
                        ob_candle["mitigation_time"] = idx
                        break

            order_blocks.append(ob_candle)

    logger.debug(
        f"Order block detection: found {len(order_blocks)} "
        f"({sum(1 for ob in order_blocks if not ob['mitigated'])} unmitigated)"
    )
    return order_blocks


# ──────────────────────────────────────────────────────────────
# 5. FAIR VALUE GAP (FVG) DETECTION
# ──────────────────────────────────────────────────────────────

def detect_fvg(
    df: pd.DataFrame,
    min_gap_pips: float = 1.0,
    pip_size: float = 0.01,
    mitigation_df: Optional[pd.DataFrame] = None
) -> list[dict]:
    """
    Detect Fair Value Gaps (FVGs) - 3-candle imbalance patterns.

    Bullish FVG: Candle 3's low > Candle 1's high (gap between them).
    Bearish FVG: Candle 1's low > Candle 3's high.

    Args:
        df: DataFrame with OHLC and DatetimeIndex.
        min_gap_pips: Minimum gap size in pips to qualify as FVG.
        pip_size: Dollar value of one pip (0.01 for XAUUSD).
        mitigation_df: Optional higher-precision DataFrame to check FVG mitigation against.

    Returns:
        List of FVG dicts.
    """
    fvgs = []
    min_gap = min_gap_pips * pip_size

    highs = df["high"].values
    lows = df["low"].values
    times = df.index

    for i in range(2, len(df)):
        c0_high = highs[i - 2]
        c0_low = lows[i - 2]
        c2_high = highs[i]
        c2_low = lows[i]

        # Bullish FVG: gap between candle 0's high and candle 2's low
        if c2_low > c0_high:
            gap = c2_low - c0_high
            if gap >= min_gap:
                fvg = {
                    "type": "FVG",
                    "direction": "bullish",
                    "top": float(c2_low),
                    "bottom": float(c0_high),
                    "time": times[i - 1],
                    "candle_0_time": times[i - 2],
                    "candle_2_time": times[i],
                    "gap_pips": round(gap / pip_size, 1),
                    "mitigated": False,
                    "mitigation_time": None,
                }
                fvgs.append(fvg)

        # Bearish FVG
        elif c0_low > c2_high:
            gap = c0_low - c2_high
            if gap >= min_gap:
                fvg = {
                    "type": "FVG",
                    "direction": "bearish",
                    "top": float(c0_low),
                    "bottom": float(c2_high),
                    "time": times[i - 1],
                    "candle_0_time": times[i - 2],
                    "candle_2_time": times[i],
                    "gap_pips": round(gap / pip_size, 1),
                    "mitigated": False,
                    "mitigation_time": None,
                }
                fvgs.append(fvg)

    # Check mitigation
    check_df = mitigation_df if mitigation_df is not None else df
    for fvg in fvgs:
        bars_after = check_df.loc[check_df.index > fvg["candle_2_time"]]
        for idx, bar in bars_after.iterrows():
            if fvg["direction"] == "bullish":
                if bar["low"] <= fvg["top"]:
                    fvg["mitigated"] = True
                    fvg["mitigation_time"] = idx
                    break
            else:
                if bar["high"] >= fvg["bottom"]:
                    fvg["mitigated"] = True
                    fvg["mitigation_time"] = idx
                    break

    logger.debug(
        f"FVG detection: found {len(fvgs)} "
        f"({sum(1 for f in fvgs if not f['mitigated'])} unmitigated)"
    )
    return fvgs


# ──────────────────────────────────────────────────────────────
# 6. LIQUIDITY SWEEP DETECTION
# ──────────────────────────────────────────────────────────────

def detect_liquidity_sweeps(
    df: pd.DataFrame,
    swings: list[dict],
    reversal_bars: int = 3
) -> list[dict]:
    """
    Detect liquidity sweeps - wicks beyond active prior swing highs/lows
    followed by a reversal. Swings are deactivated once price closes past
    them or they are swept.

    Args:
        df: DataFrame with OHLC and DatetimeIndex.
        swings: Sorted list of swing points from get_swing_list().
        reversal_bars: Number of bars after the sweep to confirm reversal.

    Returns:
        List of sweep dicts.
    """
    sweeps = []

    # Track activity of swings
    active_swings = [dict(s, active=True) for s in swings]

    for i in range(len(df)):
        bar_time = df.index[i]
        bar = df.iloc[i]

        for s in active_swings:
            if not s["active"]:
                continue
            if s["time"] >= bar_time:
                continue

            if s["type"] == "high":
                # Close above invalidates the swing high
                if bar["close"] > s["price"]:
                    s["active"] = False
                # Wick above but close below sweeps the swing high
                elif bar["high"] > s["price"] and bar["close"] < s["price"]:
                    future_bars = df.iloc[i + 1:i + 1 + reversal_bars]
                    reversal = False
                    if not future_bars.empty:
                        reversal = any(future_bars["close"] < bar["low"])

                    sweeps.append({
                        "type": "sweep",
                        "direction": "bearish",
                        "sweep_price": float(bar["high"]),
                        "sweep_time": bar_time,
                        "swing_ref": {k: v for k, v in s.items() if k != "active"},
                        "reversal_confirmed": reversal,
                    })
                    # Swings are invalidated after a sweep
                    s["active"] = False
            else:  # type == 'low'
                # Close below invalidates the swing low
                if bar["close"] < s["price"]:
                    s["active"] = False
                # Wick below but close above sweeps the swing low
                elif bar["low"] < s["price"] and bar["close"] > s["price"]:
                    future_bars = df.iloc[i + 1:i + 1 + reversal_bars]
                    reversal = False
                    if not future_bars.empty:
                        reversal = any(future_bars["close"] > bar["high"])

                    sweeps.append({
                        "type": "sweep",
                        "direction": "bullish",
                        "sweep_price": float(bar["low"]),
                        "sweep_time": bar_time,
                        "swing_ref": {k: v for k, v in s.items() if k != "active"},
                        "reversal_confirmed": reversal,
                    })
                    # Swings are invalidated after a sweep
                    s["active"] = False

    logger.debug(
        f"Liquidity sweep detection: found {len(sweeps)} "
        f"({sum(1 for s in sweeps if s['reversal_confirmed'])} confirmed)"
    )
    return sweeps


# ──────────────────────────────────────────────────────────────
# 7. STRUCTURAL BIAS INFERENCE
# ──────────────────────────────────────────────────────────────

def infer_bias(
    bos_events: list[dict],
    choch_events: list[dict]
) -> str:
    """
    Infer the current market bias from the most recent structural events.

    Returns:
        'bullish', 'bearish', or 'neutral'.
    """
    all_events = []

    for bos in bos_events:
        all_events.append({
            "type": "BOS",
            "direction": bos["direction"],
            "time": bos["break_time"],
        })

    for choch in choch_events:
        all_events.append({
            "type": "CHoCH",
            "direction": choch["direction"],
            "time": choch["break_time"],
        })

    if not all_events:
        return "neutral"

    all_events.sort(key=lambda e: e["time"])
    latest = all_events[-1]

    return latest["direction"]


# ──────────────────────────────────────────────────────────────
# 8. MASTER ANALYSIS FUNCTION
# ──────────────────────────────────────────────────────────────

def analyze_structure(
    df: pd.DataFrame,
    config: dict
) -> dict:
    """
    Run the complete SMC analysis pipeline on a DataFrame.

    Args:
        df: DataFrame with OHLC columns and UTC DatetimeIndex.
        config: The 'smc' section of the YAML config.

    Returns:
        Dict with all SMC analysis results.
    """
    smc_cfg = config if "swing_lookback" in config else config.get("smc", config)

    lookback = smc_cfg.get("swing_lookback", 5)
    ob_lookback = smc_cfg.get("ob_lookback", 10)
    fvg_min = smc_cfg.get("fvg_min_gap_pips", 1.0)
    reversal_bars = smc_cfg.get("liquidity_reversal_bars", 3)
    pip_size = smc_cfg.get("pip_size", 0.01)

    logger.info(
        f"Running SMC analysis: {len(df)} bars, "
        f"lookback={lookback}, pip_size={pip_size}"
    )

    swing_df = detect_swing_points(df, lookback=lookback)
    swings = get_swing_list(swing_df)

    bos = detect_bos(df, swings)
    choch = detect_choch(df, swings)

    order_blocks = detect_order_blocks(df, bos, ob_lookback=ob_lookback)
    
    # Resample M1 to M30 specifically for FVG detection
    df_m30 = df.resample("30min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()
    
    if len(df_m30) >= 3:
        fvgs = detect_fvg(df_m30, min_gap_pips=fvg_min, pip_size=pip_size, mitigation_df=df)
    else:
        fvgs = []
        
    sweeps = detect_liquidity_sweeps(df, swings, reversal_bars=reversal_bars)

    bias = infer_bias(bos, choch)

    logger.info(
        f"SMC analysis complete - Bias: {bias}, "
        f"BOS: {len(bos)}, CHoCH: {len(choch)}, "
        f"OBs: {len(order_blocks)}, FVGs: {len(fvgs)}, "
        f"Sweeps: {len(sweeps)}"
    )

    return {
        "swings": swings,
        "bos": bos,
        "choch": choch,
        "order_blocks": order_blocks,
        "fvgs": fvgs,
        "sweeps": sweeps,
        "bias": bias,
        "swing_df": swing_df,
    }
