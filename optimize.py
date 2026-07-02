import sys
from datetime import datetime, timezone, timedelta
import pandas as pd
from utils import load_config
from smc import analyze_structure
from orb import load_sessions, analyze_all_sessions
from signal_engine import load_rules, generate_signals

def simulate_trade(day_df, sig, pip_size=0.01) -> float:
    # Get bars after breakout time
    bars_after = day_df.loc[day_df.index > sig.timestamp]
    if bars_after.empty:
        return 0.0
        
    entry = sig.details.get("entry_price")
    sl = sig.sl
    tp = sig.tp
    direction = sig.direction
    
    if entry is None or sl is None or tp is None:
        return 0.0
        
    for idx, bar in bars_after.iterrows():
        if direction == "bullish":
            # Check Stop Loss first (conservative)
            if bar["low"] <= sl:
                return (sl - entry) / pip_size
            # Check Take Profit
            if bar["high"] >= tp:
                return (tp - entry) / pip_size
        else:
            # Bearish
            if bar["high"] >= sl:
                return (entry - sl) / pip_size
            if bar["low"] <= tp:
                return (entry - tp) / pip_size
                
    # If not hit by end of day, exit at final bar close
    final_close = bars_after.iloc[-1]["close"]
    if direction == "bullish":
        return (final_close - entry) / pip_size
    else:
        return (entry - final_close) / pip_size

def run_optimization():
    config = load_config("config.yaml")
    
    # Load fallback CSV data for in-memory optimization
    try:
        df = pd.read_csv("XAUUSD_M1.csv", index_col="time", parse_dates=True)
        print(f"Loaded {len(df)} bars from XAUUSD_M1.csv.")
    except Exception as e:
        print(f"Error loading XAUUSD_M1.csv: {e}")
        return

    # Split into daily dataframes to avoid database queries
    days = sorted(list(set(df.index.date)))
    daily_dfs = []
    for day in days:
        day_dt = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
        fetch_start = day_dt - timedelta(hours=6)
        fetch_end = day_dt + timedelta(hours=24)
        mask = (df.index >= fetch_start) & (df.index <= fetch_end)
        day_df = df.loc[mask]
        if not day_df.empty:
            daily_dfs.append((day_dt, day_df))
            
    print(f"Prepared {len(daily_dfs)} trading days for parameter search.")

    # Parameters to sweep
    min_scores = [2.5, 3.0, 3.5]
    fvg_min_gaps = [0.5, 1.0, 1.5]
    trigger_windows = [60, 90, 120]
    
    pip_size = config.get("smc", {}).get("pip_size", 0.01)
    sessions = load_sessions(config)
    
    # Calculate daily ADR
    daily_ranges = df.resample("D").agg({"high": "max", "low": "min"}).dropna()
    adr = float((daily_ranges["high"] - daily_ranges["low"]).mean())
    print(f"Base 5-Day ADR: {adr:.2f} ($USD)")

    best_pnl = -999999.0
    best_params = {}
    results = []

    print("\nStarting Parameter Sweep...")
    print("=" * 70)
    print(f"{'Min Score':10s} | {'FVG Min':8s} | {'Window':8s} | {'Signals':8s} | {'Win Rate':8s} | {'Net PnL (pips)':15s}")
    print("-" * 70)

    for min_score in min_scores:
        for fvg_min in fvg_min_gaps:
            for window in trigger_windows:
                # Update config dynamically
                config["signals"]["min_score"] = min_score
                config["smc"]["fvg_min_gap_pips"] = fvg_min
                config["signals"]["trigger_window_minutes"] = window
                
                rules = load_rules(config)
                
                total_signals = 0
                wins = 0
                losses = 0
                pnl_pips = 0.0
                
                for day_dt, day_df in daily_dfs:
                    # SMC structures
                    smc_result = analyze_structure(day_df, config)
                    
                    # ORB ranges
                    orb_results = analyze_all_sessions(day_df, sessions, day_dt, pip_size=pip_size)
                    
                    # Confluence signals
                    day_signals = generate_signals(
                        orb_results, 
                        smc_result, 
                        rules, 
                        min_score, 
                        config=config, 
                        adr=adr
                    )
                    
                    for sig in day_signals:
                        trade_pnl = simulate_trade(day_df, sig, pip_size)
                        pnl_pips += trade_pnl
                        total_signals += 1
                        if trade_pnl > 0:
                            wins += 1
                        else:
                            losses += 1
                            
                win_rate = (wins / total_signals * 100.0) if total_signals > 0 else 0.0
                print(f"{min_score:10.1f} | {fvg_min:8.1f} | {window:8d} | {total_signals:8d} | {win_rate:7.1f}% | {pnl_pips:15.1f}")
                
                results.append({
                    "min_score": min_score,
                    "fvg_min": fvg_min,
                    "window": window,
                    "signals": total_signals,
                    "win_rate": win_rate,
                    "pnl": pnl_pips
                })
                
                if pnl_pips > best_pnl:
                    best_pnl = pnl_pips
                    best_params = {
                        "min_score": min_score,
                        "fvg_min": fvg_min,
                        "window": window,
                        "win_rate": win_rate,
                        "signals": total_signals
                    }

    print("=" * 70)
    print("\nOPTIMIZATION SUMMARY:")
    print(f"  Best Net PnL:   {best_pnl:.1f} pips")
    print(f"  Best Min Score: {best_params.get('min_score')}")
    print(f"  Best FVG Min:   {best_params.get('fvg_min')} pips")
    print(f"  Best Window:    {best_params.get('window')} minutes")
    print(f"  Win Rate:       {best_params.get('win_rate'):.1f}% ({best_params.get('signals')} trades)")
    print("=" * 70)

if __name__ == "__main__":
    run_optimization()
