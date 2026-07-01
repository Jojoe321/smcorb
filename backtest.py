"""
backtest.py - Historical Backtesting Pipeline.

Runs the full analysis pipeline (Data -> SMC -> ORB -> Signal Engine)
against historical M1 data and produces a summary report.

Usage:
    python backtest.py
    python backtest.py --start 2025-06-01 --end 2025-06-30
"""

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from utils import load_config, setup_logging, date_range_days
from data_feed import MT5DataFeed
from orb import load_sessions, analyze_all_sessions
from smc import analyze_structure
from signal_engine import load_rules, generate_signals

logger = setup_logging()


def run_backtest(config: dict) -> dict:
    """Execute the full backtest pipeline over the configured date range."""
    bt_cfg = config.get("backtest", {})
    start_str = bt_cfg.get("start_date", "2025-06-01")
    end_str = bt_cfg.get("end_date", "2025-06-30")

    logger.info(f"Starting backtest: {start_str} --> {end_str}")

    feed = MT5DataFeed(config)
    sessions = load_sessions(config)
    rules = load_rules(config)
    min_score = config.get("signals", {}).get("min_score", 3.0)
    smc_config = config.get("smc", {})
    pip_size = smc_config.get("pip_size", 0.01)

    all_signals = []
    daily_results = []

    days = date_range_days(start_str, end_str)
    logger.info(f"Processing {len(days)} trading days...")

    for day in days:
        day_str = day.strftime("%Y-%m-%d")

        try:
            fetch_start = day - timedelta(hours=6)
            fetch_end = day + timedelta(hours=24)

            df = feed.fetch_historical_bars(fetch_start, fetch_end)

            if df.empty:
                logger.warning(f"No data for {day_str} - skipping")
                continue

            smc_result = analyze_structure(df, smc_config)
            orb_results = analyze_all_sessions(df, sessions, day, pip_size=pip_size)
            day_signals = generate_signals(orb_results, smc_result, rules, min_score)

            daily_summary = {
                "date": day_str,
                "bars": len(df),
                "bias": smc_result["bias"],
                "bos_count": len(smc_result["bos"]),
                "choch_count": len(smc_result["choch"]),
                "ob_count": len(smc_result["order_blocks"]),
                "fvg_count": len(smc_result["fvgs"]),
                "sweep_count": len(smc_result["sweeps"]),
                "signals": len(day_signals),
                "orb_breakouts": sum(
                    1 for orb in orb_results if orb.breakout_direction is not None
                ),
            }
            daily_results.append(daily_summary)

            for sig in day_signals:
                all_signals.append(sig.to_dict())

            logger.info(
                f"[{day_str}] Bias: {smc_result['bias']}, "
                f"Signals: {len(day_signals)}, "
                f"BOS: {len(smc_result['bos'])}, "
                f"CHoCH: {len(smc_result['choch'])}"
            )

        except Exception as e:
            logger.error(f"Error processing {day_str}: {e}", exc_info=True)
            daily_results.append({"date": day_str, "error": str(e)})

    summary = _build_summary(all_signals, daily_results)

    outcomes_csv = bt_cfg.get("trade_outcomes_csv")
    if outcomes_csv:
        summary["trade_stats"] = _match_trade_outcomes(all_signals, outcomes_csv)

    feed.shutdown()
    return {
        "signals": all_signals,
        "daily_results": daily_results,
        "summary": summary,
    }


def _build_summary(signals: list[dict], daily_results: list[dict]) -> dict:
    """Build aggregate statistics from the backtest results."""
    valid_days = [d for d in daily_results if "error" not in d]

    total_signals = len(signals)
    bullish = sum(1 for s in signals if s.get("direction") == "bullish")
    bearish = sum(1 for s in signals if s.get("direction") == "bearish")

    avg_score = 0.0
    if total_signals > 0:
        avg_score = sum(s.get("total_score", 0) for s in signals) / total_signals

    by_session = {}
    for s in signals:
        name = s.get("session_name", "unknown")
        by_session[name] = by_session.get(name, 0) + 1

    rule_freq = {}
    for s in signals:
        for rule in s.get("rules_triggered", []):
            rule_freq[rule] = rule_freq.get(rule, 0) + 1

    return {
        "days_processed": len(valid_days),
        "days_with_errors": len(daily_results) - len(valid_days),
        "total_signals": total_signals,
        "bullish_signals": bullish,
        "bearish_signals": bearish,
        "avg_score": round(avg_score, 2),
        "signals_by_session": by_session,
        "rule_trigger_frequency": rule_freq,
        "avg_bos_per_day": round(
            sum(d.get("bos_count", 0) for d in valid_days) / max(len(valid_days), 1), 1
        ),
        "avg_choch_per_day": round(
            sum(d.get("choch_count", 0) for d in valid_days) / max(len(valid_days), 1), 1
        ),
    }


def _match_trade_outcomes(
    signals: list[dict], outcomes_csv: str, max_match_minutes: int = 30
) -> dict:
    """Match signals to trade outcomes from a CSV file."""
    csv_path = Path(outcomes_csv)
    if not csv_path.exists():
        logger.warning(f"Trade outcomes CSV not found: {outcomes_csv}")
        return {"error": f"File not found: {outcomes_csv}"}

    outcomes = pd.read_csv(csv_path)
    outcomes["entry_time"] = pd.to_datetime(outcomes["entry_time"], utc=True)

    matched = 0
    wins = 0
    losses = 0
    total_pnl = 0.0

    for signal in signals:
        sig_time = signal.get("timestamp")
        if sig_time is None:
            continue
        if isinstance(sig_time, str):
            sig_time = pd.to_datetime(sig_time)
        sig_dir = signal.get("direction")

        for _, outcome in outcomes.iterrows():
            time_diff = abs((outcome["entry_time"] - sig_time).total_seconds() / 60)
            if time_diff <= max_match_minutes and outcome["direction"] == sig_dir:
                matched += 1
                pnl = float(outcome["pnl"])
                total_pnl += pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                break

    win_rate = wins / matched if matched > 0 else 0

    return {
        "matched_trades": matched,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 3),
        "total_pnl": round(total_pnl, 2),
        "unmatched_signals": len(signals) - matched,
    }


def print_report(results: dict) -> None:
    """Print a formatted backtest report to the console."""
    summary = results["summary"]

    print("\n" + "=" * 60)
    print("  XAUUSD SMC/ORB BACKTEST REPORT")
    print("=" * 60)

    print(f"\n  Days processed:      {summary['days_processed']}")
    print(f"  Days with errors:    {summary['days_with_errors']}")
    print(f"  Total signals:       {summary['total_signals']}")
    print(f"  Bullish signals:     {summary['bullish_signals']}")
    print(f"  Bearish signals:     {summary['bearish_signals']}")
    print(f"  Average score:       {summary['avg_score']}")

    print(f"\n  Avg BOS/day:         {summary['avg_bos_per_day']}")
    print(f"  Avg CHoCH/day:       {summary['avg_choch_per_day']}")

    if summary["signals_by_session"]:
        print("\n  Signals by Session:")
        for session, count in summary["signals_by_session"].items():
            print(f"    {session:15s}  {count}")

    if summary["rule_trigger_frequency"]:
        print("\n  Rule Trigger Frequency:")
        for rule, count in sorted(
            summary["rule_trigger_frequency"].items(),
            key=lambda x: x[1], reverse=True
        ):
            print(f"    {rule:30s}  {count}")

    if "trade_stats" in summary:
        ts = summary["trade_stats"]
        print("\n  Trade Outcome Matching:")
        print(f"    Matched trades:    {ts.get('matched_trades', 0)}")
        print(f"    Wins:              {ts.get('wins', 0)}")
        print(f"    Losses:            {ts.get('losses', 0)}")
        print(f"    Win rate:          {ts.get('win_rate', 0):.1%}")
        print(f"    Total PnL:         {ts.get('total_pnl', 0):.2f}")

    print("\n" + "=" * 60)


def export_signals_csv(signals: list[dict], output_path: str = "backtest_signals.csv"):
    """Export signals to a CSV file."""
    if not signals:
        logger.info("No signals to export.")
        return

    keys = signals[0].keys()
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(signals)

    logger.info(f"Signals exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="XAUUSD SMC/ORB Backtest Runner")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--start", default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--output", default="backtest_signals.csv", help="Output CSV path")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.start:
        config["backtest"]["start_date"] = args.start
    if args.end:
        config["backtest"]["end_date"] = args.end

    results = run_backtest(config)
    print_report(results)
    export_signals_csv(results["signals"], args.output)


if __name__ == "__main__":
    main()
