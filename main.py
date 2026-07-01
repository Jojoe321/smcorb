"""
main.py - CLI Entry Point for Live Mode.

Initializes the data feed, starts live polling, and runs the
analysis pipeline in real-time.

Usage:
    python main.py                    # Start live analysis (console output)
    python main.py --dashboard        # Also launch Streamlit dashboard
    python main.py --backtest         # Run backtest instead of live
"""

import argparse
import signal
import sys
import time
from datetime import datetime, timedelta, timezone

from utils import load_config, setup_logging
from data_feed import MT5DataFeed
from orb import load_sessions, analyze_all_sessions
from smc import analyze_structure
from signal_engine import load_rules, generate_signals

logger = setup_logging()


def live_analysis_callback(df, sessions, smc_config, rules, min_score, pip_size):
    """Callback run on each poll cycle to analyze the latest buffer."""
    if df.empty or len(df) < 50:
        return

    today = datetime.now(timezone.utc)

    try:
        smc_result = analyze_structure(df, smc_config)
        orb_results = analyze_all_sessions(df, sessions, today, pip_size=pip_size)
        signals = generate_signals(orb_results, smc_result, rules, min_score)

        if signals:
            for sig in signals:
                logger.info(f"SIGNAL: {sig.description}")

        latest = df.iloc[-1]
        logger.info(
            f"Price: {latest['close']:.2f} | "
            f"Bias: {smc_result['bias']} | "
            f"Active OBs: {sum(1 for ob in smc_result['order_blocks'] if not ob['mitigated'])} | "
            f"Active FVGs: {sum(1 for fvg in smc_result['fvgs'] if not fvg['mitigated'])} | "
            f"Signals: {len(signals)}"
        )

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)


def run_live(config: dict):
    """Start the live analysis loop."""
    logger.info("Starting XAUUSD live analysis...")

    feed = MT5DataFeed(config)
    sessions = load_sessions(config)
    rules = load_rules(config)
    min_score = config.get("signals", {}).get("min_score", 3.0)
    smc_config = config.get("smc", {})
    pip_size = smc_config.get("pip_size", 0.01)

    def on_update(df):
        live_analysis_callback(df, sessions, smc_config, rules, min_score, pip_size)

    def shutdown_handler(signum, frame):
        logger.info("Shutdown signal received...")
        feed.stop_polling()
        feed.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    feed.start_live_polling(callback=on_update)

    logger.info(
        f"Live analysis running - polling every "
        f"{config['polling']['interval_seconds']}s. Press Ctrl+C to stop."
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted - shutting down...")
        feed.stop_polling()
        feed.shutdown()


def main():
    parser = argparse.ArgumentParser(description="XAUUSD SMC/ORB Real-Time Analysis Tool")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--backtest", action="store_true", help="Run backtest mode")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.backtest:
        from backtest import run_backtest, print_report, export_signals_csv
        results = run_backtest(config)
        print_report(results)
        export_signals_csv(results["signals"])
    elif args.dashboard:
        import subprocess
        logger.info("Launching Streamlit dashboard...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--", "--config", args.config
        ])
    else:
        run_live(config)


if __name__ == "__main__":
    main()
