"""
data_feed.py - MetaTrader 5 Data Interface for XAUUSD.

Handles all interaction with the MT5 terminal:
  - Connection initialization with optional credential-based login
  - Broker timezone offset auto-detection
  - Historical M1 bar fetching via copy_rates_range
  - Tick-level data fetching via copy_ticks_range (with M1 fallback)
  - Live polling loop with configurable interval and in-memory buffer

All timestamps returned by this module are normalized to UTC.
"""

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from utils import setup_logging, to_utc

logger = setup_logging()


class MT5DataFeed:
    """
    Manages the MT5 connection and provides methods for fetching
    historical and live XAUUSD data.

    Usage:
        config = load_config()
        feed = MT5DataFeed(config)
        df = feed.fetch_historical_bars(start_dt, end_dt)
        feed.start_live_polling()
        ...
        feed.stop_polling()
        feed.shutdown()
    """

    def __init__(self, config: dict):
        """
        Initialize MT5 connection and detect broker timezone offset.

        Args:
            config: Parsed YAML config dict. Expected keys:
                - mt5.symbol: Trading symbol (e.g., "XAUUSD")
                - mt5.login: Account number (optional)
                - mt5.password: Account password (optional)
                - mt5.server: Broker server name (optional)
                - mt5.path: Path to terminal64.exe (optional)
                - polling.interval_seconds: Poll interval
                - polling.buffer_size: Max bars in memory
        """
        self.symbol = config["mt5"]["symbol"]
        self.poll_interval = config["polling"]["interval_seconds"]
        self.buffer_size = config["polling"]["buffer_size"]

        # MT5 credentials (optional - if absent, assumes already logged in)
        self._mt5_path = config["mt5"].get("path")
        self._mt5_login = config["mt5"].get("login")
        self._mt5_password = config["mt5"].get("password")
        self._mt5_server = config["mt5"].get("server")

        # Internal state
        self._buffer: Optional[pd.DataFrame] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._offset_seconds: int = 0

        # Initialize MT5 connection
        self._connect()

        # Detect broker timezone offset
        self._offset_seconds = self._detect_timezone_offset()
        logger.info(
            f"Broker timezone offset detected: {self._offset_seconds}s "
            f"({self._offset_seconds / 3600:.1f}h from UTC)"
        )

    # ──────────────────────────────────────────────────────────
    # Connection Management
    # ──────────────────────────────────────────────────────────

    def _connect(self) -> None:
        """
        Initialize the MT5 terminal connection.

        If login credentials are provided in config, they are passed
        to mt5.initialize() for automatic authentication. Otherwise,
        assumes the terminal is already running and logged in.

        Raises RuntimeError if connection fails.
        """
        # Build keyword arguments for mt5.initialize()
        init_kwargs = {}
        if self._mt5_path:
            init_kwargs["path"] = self._mt5_path
        if self._mt5_login:
            init_kwargs["login"] = int(self._mt5_login)
        if self._mt5_password:
            init_kwargs["password"] = self._mt5_password
        if self._mt5_server:
            init_kwargs["server"] = self._mt5_server

        if not mt5.initialize(**init_kwargs):
            error = mt5.last_error()
            raise RuntimeError(
                f"MT5 initialization failed: {error}. "
                f"Ensure the terminal is running and logged in."
            )

        # Verify the symbol is available
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            mt5.shutdown()
            raise RuntimeError(
                f"Symbol '{self.symbol}' not found. "
                f"Check your broker's symbol name (e.g., 'XAUUSDm', 'Gold')."
            )

        # Ensure the symbol is visible in Market Watch
        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                mt5.shutdown()
                raise RuntimeError(
                    f"Failed to select '{self.symbol}' in Market Watch."
                )

        logger.info(
            f"MT5 connected - Symbol: {self.symbol}, "
            f"Spread: {symbol_info.spread}, "
            f"Point: {symbol_info.point}"
        )

    def shutdown(self) -> None:
        """Cleanly stop polling and disconnect from MT5."""
        self.stop_polling()
        mt5.shutdown()
        logger.info("MT5 connection closed.")

    # ──────────────────────────────────────────────────────────
    # Timezone Detection
    # ──────────────────────────────────────────────────────────

    def _detect_timezone_offset(self) -> int:
        """
        Auto-detect the broker server's UTC offset.

        Compares the server time from the latest tick against the
        system's UTC time. Returns the offset in seconds.

        Some brokers report bar timestamps in UTC (offset = 0),
        while others use EET (UTC+2) or EEST (UTC+3).

        Guards against stale or invalid tick timestamps (e.g., when
        markets are closed and the demo server returns old/future
        timestamps) by clamping the offset to a sane range.

        Returns:
            Offset in seconds (e.g., 7200 for UTC+2).
        """
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logger.warning(
                "Could not fetch tick for timezone detection. "
                "Assuming UTC (offset = 0)."
            )
            return 0

        # tick.time is a Unix timestamp (int) in server time
        server_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        system_utc = datetime.now(timezone.utc)

        # The difference tells us the broker's offset
        # Round to nearest hour to avoid jitter from network latency
        diff_seconds = (server_time - system_utc).total_seconds()
        offset_hours = round(diff_seconds / 3600)

        # Sanity check: valid timezone offsets are -12 to +14 hours.
        # If outside this range, the tick timestamp is stale (e.g.,
        # market closed, demo server returning old data). Default to 0.
        if not (-12 <= offset_hours <= 14):
            logger.warning(
                f"Computed timezone offset {offset_hours}h is outside "
                f"sane range (-12 to +14). Tick may be stale. "
                f"Defaulting to UTC (offset = 0)."
            )
            return 0

        return int(offset_hours * 3600)

    def to_utc(self, server_time: datetime) -> datetime:
        """
        Convert a broker server timestamp to true UTC.

        Args:
            server_time: Datetime in broker server time (may be naive).

        Returns:
            Timezone-aware UTC datetime.
        """
        return to_utc(server_time, self._offset_seconds)

    @property
    def offset_seconds(self) -> int:
        """The detected broker server UTC offset in seconds."""
        return self._offset_seconds

    # ──────────────────────────────────────────────────────────
    # Historical Data: Bars
    # ──────────────────────────────────────────────────────────

    def fetch_historical_bars(
        self,
        start: datetime | str,
        end: datetime | str,
        timeframe: int = mt5.TIMEFRAME_M1
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC bars from MT5.

        Args:
            start: Start datetime (UTC). Can be a datetime or 'YYYY-MM-DD' string.
            end: End datetime (UTC). Can be a datetime or 'YYYY-MM-DD' string.
            timeframe: MT5 timeframe constant (default: M1).

        Returns:
            DataFrame with columns: [time, open, high, low, close, tick_volume, spread]
            Index is a UTC DatetimeIndex named 'time'.

        Raises:
            RuntimeError: If MT5 returns no data.
        """
        start_dt = self._parse_datetime(start)
        end_dt = self._parse_datetime(end)

        logger.info(
            f"Fetching {self.symbol} bars: {start_dt} --> {end_dt} "
            f"(TF={timeframe})"
        )

        # Convert UTC query datetimes to broker server timezone timestamps
        start_ts = int((start_dt + pd.Timedelta(seconds=self._offset_seconds)).timestamp())
        end_ts = int((end_dt + pd.Timedelta(seconds=self._offset_seconds)).timestamp())
        rates = mt5.copy_rates_range(self.symbol, timeframe, start_ts, end_ts)

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            raise RuntimeError(
                f"No bar data returned for {self.symbol} "
                f"({start_dt} --> {end_dt}). MT5 error: {error}"
            )

        df = pd.DataFrame(rates)

        # Convert Unix timestamps to UTC datetimes and apply offset correction
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        if self._offset_seconds != 0:
            df["time"] = df["time"] - pd.Timedelta(seconds=self._offset_seconds)

        df.set_index("time", inplace=True)

        # Keep only the columns we need
        cols = ["open", "high", "low", "close", "tick_volume", "spread"]
        df = df[[c for c in cols if c in df.columns]]

        logger.info(f"Fetched {len(df)} bars.")
        return df

    # ──────────────────────────────────────────────────────────
    # Historical Data: Ticks
    # ──────────────────────────────────────────────────────────

    def fetch_ticks(
        self,
        start: datetime | str,
        end: datetime | str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch tick-level data from MT5.

        Many brokers restrict tick history to recent days. If data is
        unavailable, returns None (caller should fall back to bars).

        Args:
            start: Start datetime (UTC).
            end: End datetime (UTC).

        Returns:
            DataFrame with columns: [time, bid, ask, last, flags]
            or None if no data available.
        """
        start_dt = self._parse_datetime(start)
        end_dt = self._parse_datetime(end)

        logger.info(f"Fetching {self.symbol} ticks: {start_dt} --> {end_dt}")

        # Convert UTC query datetimes to broker server timezone timestamps
        start_ts = int((start_dt + pd.Timedelta(seconds=self._offset_seconds)).timestamp())
        end_ts = int((end_dt + pd.Timedelta(seconds=self._offset_seconds)).timestamp())
        try:
            ticks = mt5.copy_ticks_range(
                self.symbol, start_ts, end_ts, mt5.COPY_TICKS_ALL
            )
        except Exception as e:
            logger.warning(f"Tick fetch failed: {e}. Falling back to bars.")
            return None

        if ticks is None or len(ticks) == 0:
            logger.warning(
                f"No tick data for {self.symbol} ({start_dt} --> {end_dt}). "
                f"Broker may restrict tick history."
            )
            return None

        df = pd.DataFrame(ticks)

        # tick.time is in seconds, time_msc is in milliseconds
        if "time_msc" in df.columns:
            df["time"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
        else:
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        if self._offset_seconds != 0:
            df["time"] = df["time"] - pd.Timedelta(seconds=self._offset_seconds)

        df.set_index("time", inplace=True)

        cols = ["bid", "ask", "last", "flags"]
        df = df[[c for c in cols if c in df.columns]]

        logger.info(f"Fetched {len(df)} ticks.")
        return df

    # ──────────────────────────────────────────────────────────
    # Unified Fetch (ticks with bar fallback)
    # ──────────────────────────────────────────────────────────

    def fetch_data(
        self,
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        """
        Fetch the best-resolution data available: ticks if the broker
        provides them, otherwise M1 bars.

        Args:
            start: Start datetime (UTC).
            end: End datetime (UTC).

        Returns:
            DataFrame of either tick or bar data.
        """
        ticks = self.fetch_ticks(start, end)
        if ticks is not None and len(ticks) > 0:
            return ticks

        logger.info("Tick data unavailable - falling back to M1 bars.")
        return self.fetch_historical_bars(start, end)

    # ──────────────────────────────────────────────────────────
    # Live Polling
    # ──────────────────────────────────────────────────────────

    def start_live_polling(
        self,
        callback: Optional[Callable[[pd.DataFrame], None]] = None
    ) -> None:
        """
        Start a background thread that polls MT5 for the latest bar
        and appends it to the in-memory buffer.

        Args:
            callback: Optional function called with the updated buffer
                      DataFrame after each poll cycle.
        """
        if self._poll_thread is not None and self._poll_thread.is_alive():
            logger.warning("Polling is already running.")
            return

        self._stop_event.clear()

        # Seed the buffer with recent history
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(hours=24)  # Last 24h for context
        try:
            self._buffer = self.fetch_historical_bars(start_dt, end_dt)
        except RuntimeError:
            self._buffer = pd.DataFrame(
                columns=["open", "high", "low", "close", "tick_volume", "spread"]
            )

        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            args=(callback,),
            daemon=True,
            name="MT5-Poller"
        )
        self._poll_thread.start()
        logger.info(
            f"Live polling started - interval: {self.poll_interval}s, "
            f"buffer size: {self.buffer_size}"
        )

    def stop_polling(self) -> None:
        """Signal the polling thread to stop and wait for it to exit."""
        if self._poll_thread is None or not self._poll_thread.is_alive():
            return

        self._stop_event.set()
        self._poll_thread.join(timeout=5)
        logger.info("Live polling stopped.")

    def _poll_loop(
        self,
        callback: Optional[Callable[[pd.DataFrame], None]]
    ) -> None:
        """
        Internal polling loop - runs in a background thread.

        Each cycle:
          1. Fetches the latest completed bar via copy_rates_from_pos
          2. Appends new bars to the buffer (deduplicating by index)
          3. Trims buffer to max size
          4. Calls the optional callback with the updated buffer
        """
        while not self._stop_event.is_set():
            try:
                # Fetch the 2 most recent bars (current forming + last completed)
                rates = mt5.copy_rates_from_pos(
                    self.symbol, mt5.TIMEFRAME_M1, 0, 2
                )

                if rates is not None and len(rates) > 0:
                    new_df = pd.DataFrame(rates)
                    new_df["time"] = pd.to_datetime(
                        new_df["time"], unit="s", utc=True
                    )
                    if self._offset_seconds != 0:
                        new_df["time"] = (
                            new_df["time"]
                            - pd.Timedelta(seconds=self._offset_seconds)
                        )
                    new_df.set_index("time", inplace=True)

                    cols = [
                        "open", "high", "low", "close",
                        "tick_volume", "spread"
                    ]
                    new_df = new_df[[c for c in cols if c in new_df.columns]]

                    with self._lock:
                        if self._buffer is not None and not self._buffer.empty:
                            # Combine and deduplicate (keep latest values)
                            combined = pd.concat([self._buffer, new_df])
                            combined = combined[
                                ~combined.index.duplicated(keep="last")
                            ]
                            combined.sort_index(inplace=True)

                            # Trim to buffer size
                            if len(combined) > self.buffer_size:
                                combined = combined.iloc[-self.buffer_size:]

                            self._buffer = combined
                        else:
                            self._buffer = new_df

                    # Notify downstream consumers
                    if callback is not None:
                        with self._lock:
                            callback(self._buffer.copy())

            except Exception as e:
                logger.error(f"Polling error: {e}", exc_info=True)

            # Wait for next cycle (interruptible)
            self._stop_event.wait(timeout=self.poll_interval)

    def get_buffer(self) -> pd.DataFrame:
        """
        Return a copy of the current in-memory data buffer.

        Returns:
            DataFrame copy of the buffer, or empty DataFrame if polling
            hasn't started.
        """
        with self._lock:
            if self._buffer is not None:
                return self._buffer.copy()
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "tick_volume", "spread"]
            )

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_datetime(dt: datetime | str) -> datetime:
        """
        Ensure input is a timezone-aware UTC datetime.

        Accepts:
          - datetime (naive -> assumed UTC, aware -> converted)
          - str in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format

        Returns:
            Timezone-aware UTC datetime.
        """
        if isinstance(dt, str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    parsed = datetime.strptime(dt, fmt)
                    return parsed.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            raise ValueError(f"Cannot parse datetime string: '{dt}'")

        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
