"""
utils.py — Shared utilities for the XAUUSD SMC/ORB Analysis Tool.

Provides:
  - YAML config loading with validation
  - Structured logging setup
  - Timezone helpers (UTC conversion)
"""

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml


# ──────────────────────────────────────────────────────────────
# Config Loading
# ──────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    """
    Load and return the YAML configuration file.

    Args:
        path: Path to the YAML config file (default: config.yaml in CWD).

    Returns:
        Parsed config as a nested dict.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    """
    Basic validation of required config sections.
    Raises ValueError if critical sections are missing.
    """
    required_sections = ["mt5", "polling", "sessions", "smc", "signals"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    # Validate session time format
    for session_key, session_cfg in config["sessions"].items():
        for time_field in ["start", "end"]:
            time_str = session_cfg.get(time_field, "")
            try:
                _parse_time_str(time_str)
            except ValueError:
                raise ValueError(
                    f"Invalid time format '{time_str}' in session '{session_key}.{time_field}'. "
                    f"Expected 'HH:MM' (24-hour UTC)."
                )


# ──────────────────────────────────────────────────────────────
# Time Helpers
# ──────────────────────────────────────────────────────────────

def _parse_time_str(time_str: str) -> tuple[int, int]:
    """
    Parse a 'HH:MM' string into (hour, minute) tuple.

    Args:
        time_str: Time string in 'HH:MM' 24-hour format.

    Returns:
        Tuple of (hour, minute).

    Raises:
        ValueError: If the format is invalid.
    """
    parts = time_str.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected 'HH:MM', got '{time_str}'")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Time out of range: {time_str}")
    return hour, minute


def parse_session_time(time_str: str) -> tuple[int, int]:
    """
    Public wrapper for parsing session time strings.

    Args:
        time_str: Time string in 'HH:MM' 24-hour format.

    Returns:
        Tuple of (hour, minute).
    """
    return _parse_time_str(time_str)


def to_utc(dt: datetime, offset_seconds: int = 0) -> datetime:
    """
    Convert a datetime to UTC by subtracting a broker server offset.

    MT5 stores bar times as UTC timestamps, but some brokers apply a
    server-side offset (e.g., UTC+2 for EET). This function normalizes
    any such offset so all downstream logic operates in true UTC.

    Args:
        dt: The datetime to convert (assumed to be in server time).
        offset_seconds: The broker's UTC offset in seconds (e.g., 7200 for UTC+2).

    Returns:
        A timezone-aware datetime in UTC.
    """
    if dt.tzinfo is None:
        # Treat as server time, apply offset
        utc_dt = dt - timedelta(seconds=offset_seconds)
        return utc_dt.replace(tzinfo=timezone.utc)
    else:
        # Already timezone-aware — convert to UTC
        return dt.astimezone(timezone.utc)


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def make_utc(year: int, month: int, day: int,
             hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    """
    Create a timezone-aware UTC datetime from components.

    Args:
        year, month, day, hour, minute, second: Date/time components.

    Returns:
        Timezone-aware UTC datetime.
    """
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def date_range_days(start_str: str, end_str: str) -> list[datetime]:
    """
    Generate a list of UTC midnight datetimes for each day in the range.

    Args:
        start_str: Start date as 'YYYY-MM-DD'.
        end_str: End date as 'YYYY-MM-DD' (inclusive).

    Returns:
        List of timezone-aware UTC datetimes at midnight for each day.
    """
    start = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure and return the root application logger.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("smcorb")
    if logger.handlers:
        # Already configured — avoid duplicate handlers
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logger.level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def fetch_high_impact_news() -> list[dict]:
    """
    Fetch high-impact USD economic events from Forex Factory's weekly XML calendar feed.
    Converts EST/EDT news release times to true UTC datetimes.
    """
    import urllib.request
    import xml.etree.ElementTree as ET
    from datetime import datetime, timezone, timedelta

    url = "https://nfs.forexfactory.com/ff_calendar_thisweek.xml"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    
    events = []
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            for event in root.findall("event"):
                country_node = event.find("country")
                impact_node = event.find("impact")
                if country_node is None or impact_node is None:
                    continue
                
                country = country_node.text
                impact = impact_node.text
                
                # We only care about high-impact USD events
                if country != "USD" or impact != "High":
                    continue
                    
                title = event.find("title").text if event.find("title") is not None else "USD News Event"
                date_str = event.find("date").text
                time_str = event.find("time").text
                
                if time_str == "All Day" or not date_str or not time_str:
                    continue
                    
                # Parse date and time in Eastern Time
                try:
                    dt_str = f"{date_str} {time_str}"
                    est_dt = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p")
                    
                    # Convert Eastern Time to UTC (Daylight Savings simple approximation)
                    is_dst = 3 < est_dt.month < 11
                    offset_hours = 4 if is_dst else 5
                    utc_dt = est_dt.replace(tzinfo=timezone.utc) + timedelta(hours=offset_hours)
                    
                    events.append({
                        "title": title,
                        "time": utc_dt,
                        "impact": impact
                    })
                except Exception:
                    continue
    except Exception:
        pass
        
    return events
