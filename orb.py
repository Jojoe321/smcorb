"""
orb.py — Opening Range Breakout (ORB) Session Analysis.

Defines trading session windows (Asian, London, New York) and calculates:
  - Opening range (high/low) for the first N minutes of each session
  - Breakout detection when price closes above/below the range
  - Retest confirmation logic (optional)

All times are expected in UTC. Session boundaries are configurable via config.yaml.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, time, timedelta, timezone
from typing import Optional

import pandas as pd

from utils import parse_session_time, setup_logging

logger = setup_logging()


# ──────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────

@dataclass
class SessionConfig:
    """Configuration for a single trading session."""
    name: str
    start_hour: int
    start_minute: int
    end_hour: int
    end_minute: int
    or_minutes: int  # Opening range duration in minutes

    @property
    def start_time(self) -> time:
        """Session start as a time object."""
        return time(self.start_hour, self.start_minute)

    @property
    def end_time(self) -> time:
        """Session end as a time object."""
        return time(self.end_hour, self.end_minute)


@dataclass
class ORBResult:
    """
    Result of opening range breakout analysis for a single session.

    This is the primary output structure — one per session per day.
    """
    session_name: str
    date: str                              # 'YYYY-MM-DD'
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    range_open_time: Optional[datetime] = None
    range_close_time: Optional[datetime] = None
    breakout_direction: Optional[str] = None   # 'bullish', 'bearish', or None
    breakout_time: Optional[datetime] = None
    breakout_price: Optional[float] = None
    retest_confirmed: bool = False
    retest_time: Optional[datetime] = None
    bars_in_range: int = 0

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        d = asdict(self)
        # Convert datetimes to ISO strings for JSON compatibility
        for key in ["range_open_time", "range_close_time",
                     "breakout_time", "retest_time"]:
            if d[key] is not None:
                d[key] = d[key].isoformat()
        return d


# ──────────────────────────────────────────────────────────────
# Config Parsing
# ──────────────────────────────────────────────────────────────

def load_sessions(config: dict) -> list[SessionConfig]:
    """
    Parse session definitions from the YAML config.

    Args:
        config: Parsed config dict with a 'sessions' key.

    Returns:
        List of SessionConfig objects, one per defined session.
    """
    sessions = []
    for key, sess_cfg in config["sessions"].items():
        start_h, start_m = parse_session_time(sess_cfg["start"])
        end_h, end_m = parse_session_time(sess_cfg["end"])
        sessions.append(SessionConfig(
            name=sess_cfg.get("name", key.capitalize()),
            start_hour=start_h,
            start_minute=start_m,
            end_hour=end_h,
            end_minute=end_m,
            or_minutes=sess_cfg.get("or_minutes", 15),
        ))
    return sessions


# ──────────────────────────────────────────────────────────────
# Opening Range Calculation
# ──────────────────────────────────────────────────────────────

def calculate_opening_range(
    df: pd.DataFrame,
    session: SessionConfig,
    date: datetime
) -> dict:
    """
    Calculate the opening range (high/low) for the first N minutes
    of a trading session on a given date.

    The opening range is defined as the highest high and lowest low
    within the first `session.or_minutes` minutes after session open.

    Args:
        df: DataFrame with UTC DatetimeIndex and OHLC columns.
        session: SessionConfig defining the session window.
        date: The trading date (only the date part is used).

    Returns:
        Dict with keys: range_high, range_low, range_open_time,
        range_close_time, bars_in_range. Values are None if
        insufficient data.
    """
    # Build the opening range window
    or_start = datetime(
        date.year, date.month, date.day,
        session.start_hour, session.start_minute,
        tzinfo=timezone.utc
    )
    or_end = or_start + timedelta(minutes=session.or_minutes)

    # Filter bars within the opening range window
    mask = (df.index >= or_start) & (df.index < or_end)
    or_bars = df.loc[mask]

    if or_bars.empty:
        return {
            "range_high": None,
            "range_low": None,
            "range_open_time": or_start,
            "range_close_time": or_end,
            "bars_in_range": 0,
        }

    return {
        "range_high": float(or_bars["high"].max()),
        "range_low": float(or_bars["low"].min()),
        "range_open_time": or_start,
        "range_close_time": or_end,
        "bars_in_range": len(or_bars),
    }


# ──────────────────────────────────────────────────────────────
# Breakout Detection
# ──────────────────────────────────────────────────────────────

def detect_breakout(
    df: pd.DataFrame,
    opening_range: dict,
    session: SessionConfig,
    date: datetime
) -> dict:
    """
    Detect the first breakout of the opening range within the
    remainder of the session.

    A breakout occurs when a bar CLOSES above range_high (bullish)
    or below range_low (bearish). Using close (not high/low) reduces
    false breakouts from wicks.

    Args:
        df: DataFrame with UTC DatetimeIndex and OHLC columns.
        opening_range: Output from calculate_opening_range().
        session: SessionConfig defining the session window.
        date: The trading date.

    Returns:
        Dict with: breakout_direction, breakout_time, breakout_price.
        All None if no breakout detected.
    """
    range_high = opening_range.get("range_high")
    range_low = opening_range.get("range_low")

    if range_high is None or range_low is None:
        return {
            "breakout_direction": None,
            "breakout_time": None,
            "breakout_price": None,
        }

    # Scan bars after the opening range closes, within session bounds
    or_end = opening_range["range_close_time"]
    session_end = datetime(
        date.year, date.month, date.day,
        session.end_hour, session.end_minute,
        tzinfo=timezone.utc
    )

    # Handle sessions that cross midnight (e.g., Asian session)
    if session_end <= or_end:
        session_end += timedelta(days=1)

    mask = (df.index >= or_end) & (df.index < session_end)
    post_or_bars = df.loc[mask]

    for idx, bar in post_or_bars.iterrows():
        # Bullish breakout: close above range high
        if bar["close"] > range_high:
            return {
                "breakout_direction": "bullish",
                "breakout_time": idx,
                "breakout_price": float(bar["close"]),
            }
        # Bearish breakout: close below range low
        elif bar["close"] < range_low:
            return {
                "breakout_direction": "bearish",
                "breakout_time": idx,
                "breakout_price": float(bar["close"]),
            }

    return {
        "breakout_direction": None,
        "breakout_time": None,
        "breakout_price": None,
    }


# ──────────────────────────────────────────────────────────────
# Retest Confirmation
# ──────────────────────────────────────────────────────────────

def check_retest(
    df: pd.DataFrame,
    breakout: dict,
    opening_range: dict,
    max_bars: int = 10,
    tolerance_pips: float = 2.0,
    pip_size: float = 0.01
) -> dict:
    """
    After a breakout, check if price retests the opening range edge
    and holds (confirming the breakout).

    Retest logic:
      - Bullish breakout: price pulls back to within tolerance of
        range_high, then the next bar closes above range_high.
      - Bearish breakout: price pulls back to within tolerance of
        range_low, then the next bar closes below range_low.

    Args:
        df: DataFrame with UTC DatetimeIndex and OHLC columns.
        breakout: Output from detect_breakout().
        opening_range: Output from calculate_opening_range().
        max_bars: Max bars after breakout to look for retest.
        tolerance_pips: How close price must come to the range edge
                        to qualify as a retest.
        pip_size: Dollar value of one pip (0.01 for XAUUSD).

    Returns:
        Dict with: retest_confirmed (bool), retest_time (datetime|None).
    """
    direction = breakout.get("breakout_direction")
    breakout_time = breakout.get("breakout_time")

    if direction is None or breakout_time is None:
        return {"retest_confirmed": False, "retest_time": None}

    tolerance = tolerance_pips * pip_size

    # Get bars after the breakout bar
    post_breakout = df.loc[df.index > breakout_time].head(max_bars)

    range_high = opening_range["range_high"]
    range_low = opening_range["range_low"]

    for i, (idx, bar) in enumerate(post_breakout.iterrows()):
        if direction == "bullish":
            # Price must dip close to range_high (the support now)
            if bar["low"] <= range_high + tolerance:
                # Check if the bar (or next bar) closes above range_high
                if bar["close"] > range_high:
                    return {"retest_confirmed": True, "retest_time": idx}
                # Check next bar if available
                remaining = post_breakout.iloc[i + 1:i + 2]
                if not remaining.empty and remaining.iloc[0]["close"] > range_high:
                    return {
                        "retest_confirmed": True,
                        "retest_time": remaining.index[0],
                    }

        elif direction == "bearish":
            # Price must push back up close to range_low (resistance now)
            if bar["high"] >= range_low - tolerance:
                if bar["close"] < range_low:
                    return {"retest_confirmed": True, "retest_time": idx}
                remaining = post_breakout.iloc[i + 1:i + 2]
                if not remaining.empty and remaining.iloc[0]["close"] < range_low:
                    return {
                        "retest_confirmed": True,
                        "retest_time": remaining.index[0],
                    }

    return {"retest_confirmed": False, "retest_time": None}


# ──────────────────────────────────────────────────────────────
# Full Session Analysis Pipeline
# ──────────────────────────────────────────────────────────────

def analyze_session(
    df: pd.DataFrame,
    session: SessionConfig,
    date: datetime,
    pip_size: float = 0.01
) -> ORBResult:
    """
    Run the complete ORB analysis pipeline for a single session on
    a given date.

    Pipeline: calculate opening range → detect breakout → check retest.

    Args:
        df: DataFrame with UTC DatetimeIndex and OHLC columns.
        session: SessionConfig for the session to analyze.
        date: The trading date.
        pip_size: Dollar value of one pip (for retest tolerance).

    Returns:
        ORBResult with all fields populated.
    """
    date_str = date.strftime("%Y-%m-%d")

    # Step 1: Calculate opening range
    or_data = calculate_opening_range(df, session, date)

    # Step 2: Detect breakout
    bo_data = detect_breakout(df, or_data, session, date)

    # Step 3: Check retest (only if breakout occurred)
    rt_data = {"retest_confirmed": False, "retest_time": None}
    if bo_data["breakout_direction"] is not None:
        rt_data = check_retest(
            df, bo_data, or_data, pip_size=pip_size
        )

    return ORBResult(
        session_name=session.name,
        date=date_str,
        range_high=or_data["range_high"],
        range_low=or_data["range_low"],
        range_open_time=or_data["range_open_time"],
        range_close_time=or_data["range_close_time"],
        breakout_direction=bo_data["breakout_direction"],
        breakout_time=bo_data["breakout_time"],
        breakout_price=bo_data["breakout_price"],
        retest_confirmed=rt_data["retest_confirmed"],
        retest_time=rt_data["retest_time"],
        bars_in_range=or_data["bars_in_range"],
    )


def analyze_all_sessions(
    df: pd.DataFrame,
    sessions: list[SessionConfig],
    date: datetime,
    pip_size: float = 0.01
) -> list[ORBResult]:
    """
    Run ORB analysis for all configured sessions on a given date.

    Args:
        df: DataFrame with UTC DatetimeIndex and OHLC columns.
        sessions: List of SessionConfig objects.
        date: The trading date.
        pip_size: Dollar value of one pip.

    Returns:
        List of ORBResult objects, one per session.
    """
    results = []
    for session in sessions:
        result = analyze_session(df, session, date, pip_size)
        results.append(result)
        logger.debug(
            f"ORB [{session.name}] {date.strftime('%Y-%m-%d')}: "
            f"H={result.range_high} L={result.range_low} "
            f"BO={result.breakout_direction}"
        )
    return results
