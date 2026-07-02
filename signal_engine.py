"""
signal_engine.py - Confluence Signal Combiner.

Merges SMC structural analysis with ORB session data to identify
high-probability trading setups. Uses configurable weighted rules
to score confluence between different factors.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Optional

from utils import setup_logging

logger = setup_logging()


# ──────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────

@dataclass
class ConfluenceRule:
    """A single confluence rule for signal scoring."""
    name: str
    weight: float
    description: str
    evaluate: Optional[Callable] = None


@dataclass
class Signal:
    """A scored trading signal combining ORB and SMC analysis."""
    session_name: str
    date: str
    direction: str
    total_score: float
    max_possible_score: float
    score_pct: float
    rules_triggered: list[str]
    rules_missed: list[str]
    details: dict
    timestamp: Optional[datetime] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    rr_ratio: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["timestamp"] is not None:
            d["timestamp"] = d["timestamp"].isoformat()
        return d

    @property
    def description(self) -> str:
        triggered = ", ".join(self.rules_triggered) or "none"
        return (
            f"{self.session_name} ORB {self.direction} breakout | "
            f"Score: {self.total_score:.1f}/{self.max_possible_score:.1f} "
            f"({self.score_pct:.0%}) | Confluence: {triggered}"
        )


# ──────────────────────────────────────────────────────────────
# Default Rule Implementations
# ──────────────────────────────────────────────────────────────

def _rule_orb_aligned_with_bias(orb_result, smc_result) -> bool:
    """ORB breakout direction matches the current SMC structural bias."""
    if orb_result.breakout_direction is None:
        return False
    return orb_result.breakout_direction == smc_result.get("bias")


def _rule_unmitigated_ob(orb_result, smc_result) -> bool:
    """An unmitigated order block exists in the breakout direction."""
    if orb_result.breakout_direction is None:
        return False
    for ob in smc_result.get("order_blocks", []):
        if not ob["mitigated"] and ob["direction"] == orb_result.breakout_direction:
            return True
    return False


def _rule_fvg_confluence(orb_result, smc_result) -> bool:
    """An unmitigated FVG aligns with the breakout direction."""
    if orb_result.breakout_direction is None:
        return False
    for fvg in smc_result.get("fvgs", []):
        if not fvg["mitigated"] and fvg["direction"] == orb_result.breakout_direction:
            return True
    return False


def _rule_liquidity_sweep_before(orb_result, smc_result) -> bool:
    """A confirmed liquidity sweep on the opposite side preceded the breakout."""
    if orb_result.breakout_direction is None or orb_result.breakout_time is None:
        return False
    for sweep in smc_result.get("sweeps", []):
        if (sweep["reversal_confirmed"]
                and sweep["direction"] == orb_result.breakout_direction
                and sweep["sweep_time"] < orb_result.breakout_time):
            return True
    return False


def _rule_retest_confirmed(orb_result, smc_result) -> bool:
    """The ORB breakout was followed by a successful retest."""
    return orb_result.retest_confirmed


def _rule_macro_trend_aligned(orb_result, smc_result) -> bool:
    """ORB breakout direction matches the M30 macro trend bias."""
    if orb_result.breakout_direction is None:
        return False
    return orb_result.breakout_direction == smc_result.get("macro_bias")


_DEFAULT_RULE_FNS = {
    "orb_aligned_with_bias": _rule_orb_aligned_with_bias,
    "macro_trend_aligned": _rule_macro_trend_aligned,
    "unmitigated_ob": _rule_unmitigated_ob,
    "fvg_confluence": _rule_fvg_confluence,
    "liquidity_sweep_before": _rule_liquidity_sweep_before,
    "retest_confirmed": _rule_retest_confirmed,
}


# ──────────────────────────────────────────────────────────────
# Rule Loading
# ──────────────────────────────────────────────────────────────

def load_rules(config: dict) -> list[ConfluenceRule]:
    """Load confluence rules from the YAML config."""
    rules = []
    rules_cfg = config.get("signals", {}).get("rules", {})

    for name, rule_cfg in rules_cfg.items():
        fn = _DEFAULT_RULE_FNS.get(name)
        if fn is None:
            logger.warning(
                f"Rule '{name}' has no evaluation function - "
                f"it will never trigger."
            )

        rules.append(ConfluenceRule(
            name=name,
            weight=rule_cfg.get("weight", 1.0),
            description=rule_cfg.get("description", ""),
            evaluate=fn,
        ))

    return rules


# ──────────────────────────────────────────────────────────────
# Signal Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_confluence(
    orb_result,
    smc_result: dict,
    rules: list[ConfluenceRule],
    min_score: float = 3.0,
    config: Optional[dict] = None,
    adr: Optional[float] = None,
    news_events: Optional[list[dict]] = None
) -> Optional[Signal]:
    """Score a single ORB result against SMC analysis using confluence rules."""
    if orb_result.breakout_direction is None:
        return None

    # Load configuration settings if available
    sig_cfg = config.get("signals", {}) if config else {}
    from datetime import timedelta

    # ── Time-of-day execution filter ──
    trigger_window = sig_cfg.get("trigger_window_minutes", 90)
    if orb_result.breakout_time and orb_result.range_close_time:
        session_start = orb_result.range_close_time - timedelta(minutes=30)
        breakout_delay = (orb_result.breakout_time - session_start).total_seconds() / 60.0
        if breakout_delay > trigger_window:
            logger.info(
                f"[{orb_result.session_name}] Breakout at {orb_result.breakout_time.strftime('%H:%M')} "
                f"({breakout_delay:.1f}m delay) is outside trigger window of {trigger_window}m. Signal blocked."
            )
            return None

    # ── ADR filter ──
    adr_filter_pct = sig_cfg.get("adr_filter_pct", 80.0)
    if adr is not None and orb_result.range_high is not None and orb_result.range_low is not None:
        today_range = float(orb_result.range_high - orb_result.range_low)
        adr_limit = adr * (adr_filter_pct / 100.0)
        if today_range > adr_limit:
            logger.info(
                f"[{orb_result.session_name}] Session range ({today_range:.2f}) exceeds ADR limit of "
                f"{adr_limit:.2f} ({adr_filter_pct}% of {adr:.2f} ADR). Signal blocked."
            )
            return None

    # ── Economic News filter ──
    if sig_cfg.get("news_filter_enabled", True) and news_events and orb_result.breakout_time:
        buffer_mins = sig_cfg.get("news_buffer_minutes", 30)
        for event in news_events:
            event_time = event["time"]
            time_diff = abs((orb_result.breakout_time - event_time).total_seconds()) / 60.0
            if time_diff <= buffer_mins:
                logger.info(
                    f"[{orb_result.session_name}] Breakout at {orb_result.breakout_time.strftime('%H:%M')} "
                    f"is within {buffer_mins}m of news event '{event['title']}' at {event_time.strftime('%H:%M')}. Signal blocked."
                )
                return None

    total_score = 0.0
    max_score = sum(r.weight for r in rules)
    triggered = []
    missed = []
    details = {}

    for rule in rules:
        if rule.evaluate is None:
            missed.append(rule.name)
            continue

        try:
            if rule.evaluate(orb_result, smc_result):
                total_score += rule.weight
                triggered.append(rule.name)
                details[rule.name] = {
                    "weight": rule.weight,
                    "triggered": True,
                    "description": rule.description,
                }
            else:
                missed.append(rule.name)
                details[rule.name] = {
                    "weight": rule.weight,
                    "triggered": False,
                    "description": rule.description,
                }
        except Exception as e:
            logger.error(f"Rule '{rule.name}' evaluation error: {e}")
            missed.append(rule.name)

    details["smc_bias"] = smc_result.get("bias", "neutral")
    details["unmitigated_obs"] = sum(
        1 for ob in smc_result.get("order_blocks", [])
        if not ob["mitigated"]
    )
    details["unmitigated_fvgs"] = sum(
        1 for fvg in smc_result.get("fvgs", [])
        if not fvg["mitigated"]
    )

    if total_score < min_score:
        logger.debug(
            f"[{orb_result.session_name}] Score {total_score:.1f} < "
            f"min {min_score:.1f} - no signal"
        )
        return None

    # Calculate Dynamic Stop Loss & Take Profit Levels
    entry = float(orb_result.breakout_price)
    sl = 0.0
    tp = 0.0
    
    if orb_result.breakout_direction == "bullish":
        sl = float(orb_result.range_low) if orb_result.range_low is not None else entry * 0.99
        targets = []
        for ob in smc_result.get("order_blocks", []):
            if not ob["mitigated"] and ob["direction"] == "bearish" and ob["top"] > entry:
                targets.append(ob["top"])
        for fvg in smc_result.get("fvgs", []):
            if not fvg["mitigated"] and fvg["direction"] == "bearish" and fvg["bottom"] > entry:
                targets.append(fvg["bottom"])
        if targets:
            tp = float(min(targets))
        else:
            tp = entry + 2.0 * (entry - sl)
    else:
        sl = float(orb_result.range_high) if orb_result.range_high is not None else entry * 1.01
        targets = []
        for ob in smc_result.get("order_blocks", []):
            if not ob["mitigated"] and ob["direction"] == "bullish" and ob["bottom"] < entry:
                targets.append(ob["bottom"])
        for fvg in smc_result.get("fvgs", []):
            if not fvg["mitigated"] and fvg["direction"] == "bullish" and fvg["top"] < entry:
                targets.append(fvg["top"])
        if targets:
            tp = float(max(targets))
        else:
            tp = entry - 2.0 * (sl - entry)
            
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

    details["entry_price"] = round(entry, 2)

    signal = Signal(
        session_name=orb_result.session_name,
        date=orb_result.date,
        direction=orb_result.breakout_direction,
        total_score=total_score,
        max_possible_score=max_score,
        score_pct=total_score / max_score if max_score > 0 else 0,
        rules_triggered=triggered,
        rules_missed=missed,
        details=details,
        timestamp=orb_result.breakout_time,
        sl=round(sl, 2),
        tp=round(tp, 2),
        rr_ratio=rr_ratio,
    )

    logger.info(f"Signal generated: {signal.description} | SL: {signal.sl} | TP: {signal.tp} (R:R: {signal.rr_ratio})")
    return signal


def generate_signals(
    orb_results: list,
    smc_result: dict,
    rules: list[ConfluenceRule],
    min_score: float = 3.0,
    config: Optional[dict] = None,
    adr: Optional[float] = None,
    news_events: Optional[list[dict]] = None
) -> list[Signal]:
    """Generate signals for all ORB results against the SMC analysis."""
    signals = []
    for orb in orb_results:
        signal = evaluate_confluence(orb, smc_result, rules, min_score, config, adr, news_events)
        if signal is not None:
            signals.append(signal)

    logger.info(
        f"Signal generation: {len(signals)} signals from "
        f"{len(orb_results)} sessions"
    )
    return signals
