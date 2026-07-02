"""
dashboard.py — Streamlit Live Dashboard for XAUUSD SMC/ORB Analysis.

Displays a Plotly candlestick chart with overlaid:
  - Session ranges (colored rectangles)
  - Order blocks (shaded zones)
  - Fair Value Gaps (semi-transparent rectangles)
  - BOS/CHoCH markers (annotated arrows)
  - Breakout signals (triangle markers)

Auto-refreshes from the live polling loop.

Usage:
    streamlit run dashboard.py
    streamlit run dashboard.py -- --config path/to/config.yaml
"""

import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import load_config, setup_logging
from data_feed import MT5DataFeed
from orb import load_sessions, analyze_all_sessions, ORBResult
from smc import analyze_structure
from signal_engine import load_rules, generate_signals, Signal


logger = setup_logging()

# ──────────────────────────────────────────────────────────────
# Streamlit Page Config
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="XAUUSD SMC/ORB Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────────────────────

@st.cache_resource
def init_data_feed(config_path: str):
    """Initialize MT5 data feed (cached across reruns)."""
    config = load_config(config_path)
    feed = MT5DataFeed(config)
    return feed, config


def get_config_path() -> str:
    """Get config path from CLI args or default."""
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "config.yaml"


# ──────────────────────────────────────────────────────────────
# Chart Building
# ──────────────────────────────────────────────────────────────

def build_candlestick_chart(
    df: pd.DataFrame,
    orb_results: list[ORBResult],
    smc_result: dict,
    signals: list[Signal],
    chart_bars: int = 500,
) -> go.Figure:
    """
    Build the main Plotly candlestick chart with all overlays.

    Args:
        df: OHLC DataFrame with UTC DatetimeIndex.
        orb_results: List of ORBResult objects for the current day.
        smc_result: Dict from analyze_structure().
        signals: List of Signal objects.
        chart_bars: Number of recent bars to display.

    Returns:
        Plotly Figure object.
    """
    # Trim to most recent bars
    plot_df = df.tail(chart_bars).copy()

    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting for data...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="#888")
        )
        return fig

    fig = go.Figure()

    # ── Candlestick chart ──
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["open"],
        high=plot_df["high"],
        low=plot_df["low"],
        close=plot_df["close"],
        name="XAUUSD",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
    ))

    # ── Session Range Rectangles ──
    session_colors = {
        "Asian": "rgba(66, 133, 244, 0.15)",      # Blue
        "London": "rgba(52, 168, 83, 0.15)",       # Green
        "New York": "rgba(251, 188, 4, 0.15)",     # Orange
    }
    session_border_colors = {
        "Asian": "rgba(66, 133, 244, 0.6)",
        "London": "rgba(52, 168, 83, 0.6)",
        "New York": "rgba(251, 188, 4, 0.6)",
    }

    for orb in orb_results:
        if orb.range_high is None or orb.range_low is None:
            continue

        fill_color = session_colors.get(
            orb.session_name, "rgba(158, 158, 158, 0.08)"
        )
        border_color = session_border_colors.get(
            orb.session_name, "rgba(158, 158, 158, 0.4)"
        )

        # Opening range rectangle
        fig.add_shape(
            type="rect",
            x0=orb.range_open_time,
            x1=orb.range_close_time,
            y0=orb.range_low,
            y1=orb.range_high,
            fillcolor=fill_color,
            line=dict(color=border_color, width=1, dash="dash"),
            name=f"{orb.session_name} OR",
        )

        # Range high/low horizontal lines (extend through session)
        fig.add_shape(
            type="line",
            x0=orb.range_open_time,
            x1=plot_df.index[-1],
            y0=orb.range_high,
            y1=orb.range_high,
            line=dict(color=border_color, width=1, dash="dot"),
        )
        fig.add_shape(
            type="line",
            x0=orb.range_open_time,
            x1=plot_df.index[-1],
            y0=orb.range_low,
            y1=orb.range_low,
            line=dict(color=border_color, width=1, dash="dot"),
        )

    # ── Order Blocks ──
    for ob in smc_result.get("order_blocks", []):
        if ob["ob_time"] < plot_df.index[0]:
            continue

        if ob["mitigated"]:
            color = "rgba(158, 158, 158, 0.12)"  # Gray for mitigated
            border = "rgba(158, 158, 158, 0.3)"
        elif ob["direction"] == "bullish":
            color = "rgba(76, 175, 80, 0.15)"    # Green for demand
            border = "rgba(76, 175, 80, 0.5)"
        else:
            color = "rgba(244, 67, 54, 0.15)"    # Red for supply
            border = "rgba(244, 67, 54, 0.5)"

        # OB extends from creation time to mitigation or chart end
        x1 = ob.get("mitigation_time", plot_df.index[-1])
        if x1 is None:
            x1 = plot_df.index[-1]

        fig.add_shape(
            type="rect",
            x0=ob["ob_time"],
            x1=x1,
            y0=ob["bottom"],
            y1=ob["top"],
            fillcolor=color,
            line=dict(color=border, width=1),
        )

    # ── Fair Value Gaps ──
    for fvg in smc_result.get("fvgs", []):
        if fvg["time"] < plot_df.index[0]:
            continue

        if fvg["mitigated"]:
            color = "rgba(158, 158, 158, 0.08)"
        elif fvg["direction"] == "bullish":
            color = "rgba(0, 150, 136, 0.12)"
        else:
            color = "rgba(233, 30, 99, 0.12)"

        x1 = fvg.get("mitigation_time", plot_df.index[-1])
        if x1 is None:
            x1 = plot_df.index[-1]

        fig.add_shape(
            type="rect",
            x0=fvg["time"],
            x1=x1,
            y0=fvg["bottom"],
            y1=fvg["top"],
            fillcolor=color,
            line=dict(width=0),
        )

    # ── Swing Points ──
    swing_df = smc_result.get("swing_df")
    if swing_df is not None:
        swing_df_plot = swing_df.loc[swing_df.index >= plot_df.index[0]]

        sh_mask = swing_df_plot["swing_high"] == True
        if sh_mask.any():
            fig.add_trace(go.Scatter(
                x=swing_df_plot.loc[sh_mask].index,
                y=swing_df_plot.loc[sh_mask]["high"],
                mode="markers",
                marker=dict(
                    symbol="triangle-down", size=8,
                    color="#ef5350", line=dict(width=1, color="white")
                ),
                name="Swing High",
                showlegend=True,
            ))

        sl_mask = swing_df_plot["swing_low"] == True
        if sl_mask.any():
            fig.add_trace(go.Scatter(
                x=swing_df_plot.loc[sl_mask].index,
                y=swing_df_plot.loc[sl_mask]["low"],
                mode="markers",
                marker=dict(
                    symbol="triangle-up", size=8,
                    color="#26a69a", line=dict(width=1, color="white")
                ),
                name="Swing Low",
                showlegend=True,
            ))

    # ── BOS / CHoCH Markers ──
    for bos in smc_result.get("bos", []):
        if bos["break_time"] < plot_df.index[0]:
            continue
        color = "#26a69a" if bos["direction"] == "bullish" else "#ef5350"
        fig.add_annotation(
            x=bos["break_time"],
            y=bos["break_price"],
            text=f"BOS {'↑' if bos['direction'] == 'bullish' else '↓'}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=color,
            font=dict(size=9, color=color, family="monospace"),
            bgcolor="rgba(30, 30, 30, 0.7)",
            bordercolor=color,
            borderwidth=1,
        )

    for choch in smc_result.get("choch", []):
        if choch["break_time"] < plot_df.index[0]:
            continue
        color = "#ff9800" if choch["direction"] == "bullish" else "#9c27b0"
        fig.add_annotation(
            x=choch["break_time"],
            y=choch["break_price"],
            text=f"CHoCH {'↑' if choch['direction'] == 'bullish' else '↓'}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=color,
            font=dict(size=9, color=color, family="monospace"),
            bgcolor="rgba(30, 30, 30, 0.7)",
            bordercolor=color,
            borderwidth=1,
        )

    # ── Breakout Signals ──
    for sig in signals:
        if sig.timestamp is None:
            continue
        if sig.timestamp < plot_df.index[0]:
            continue

        color = "#00e676" if sig.direction == "bullish" else "#ff1744"
        symbol = "triangle-up" if sig.direction == "bullish" else "triangle-down"

        # Get the price at signal time
        if sig.timestamp in plot_df.index:
            price = plot_df.loc[sig.timestamp, "close"]
        else:
            price = plot_df.iloc[-1]["close"]

        fig.add_trace(go.Scatter(
            x=[sig.timestamp],
            y=[price],
            mode="markers+text",
            marker=dict(symbol=symbol, size=14, color=color),
            text=[f"{sig.total_score:.0f}"],
            textposition="top center",
            textfont=dict(size=10, color=color),
            name=f"Signal ({sig.session_name})",
            showlegend=False,
        ))

    # ── Layout ──
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text="XAUUSD — Smart Money Concepts + Opening Range Breakout",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Time (UTC)",
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis=dict(
            title="Price",
            side="right",
        ),
        height=700,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        hovermode="x unified",
    )

    return fig


# ──────────────────────────────────────────────────────────────
# Page Renderers
# ──────────────────────────────────────────────────────────────

def render_main_dashboard(
    df: pd.DataFrame,
    orb_results: list[ORBResult],
    smc_result: dict,
    signals: list[Signal],
    chart_bars: int
):
    """Render the primary candlestick chart and metrics."""
    # ── Render Candlestick Chart ──
    fig = build_candlestick_chart(
        df, orb_results, smc_result, signals, chart_bars
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ── Quick Overview Panel ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        bias = smc_result.get("bias", "neutral")
        bias_color = (
            "🟢" if bias == "bullish"
            else "🔴" if bias == "bearish"
            else "⚪"
        )
        st.metric("SMC Bias", f"{bias_color} {bias.upper()}")

    with col2:
        active_obs = sum(
            1 for ob in smc_result.get("order_blocks", [])
            if not ob["mitigated"]
        )
        st.metric("Active Order Blocks", active_obs)

    with col3:
        active_fvgs = sum(
            1 for fvg in smc_result.get("fvgs", [])
            if not fvg["mitigated"]
        )
        st.metric("Active FVGs", active_fvgs)

    with col4:
        st.metric("Signals", len(signals))


def render_session_details(orb_results: list[ORBResult], latest_bar_time: datetime, sessions: list[SessionConfig], adr: Optional[float] = None):
    """Render a premium grid card layout for session ORB details with clear progress tracking."""
    st.markdown("## 📋 Session ORB Details")
    
    if adr is not None:
        highs = [orb.range_high for orb in orb_results if orb.range_high is not None]
        lows = [orb.range_low for orb in orb_results if orb.range_low is not None]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("5-Day ADR", f"${adr:.2f}")
        if highs and lows:
            today_high = max(highs)
            today_low = min(lows)
            today_range = today_high - today_low
            adr_pct = (today_range / adr) * 100.0
            with col2:
                st.metric("Today's Session Range", f"${today_range:.2f}")
            with col3:
                st.metric("ADR Coverage", f"{adr_pct:.1f}%")
        else:
            with col2:
                st.metric("Today's Session Range", "N/A")
            with col3:
                st.metric("ADR Coverage", "0.0%")
        st.divider()
    
    if not orb_results:
        st.info("No session details available for this day/range.")
        return
        
    sess_map = {s.name: s for s in sessions}
        
    for orb in orb_results:
        session_cfg = sess_map.get(orb.session_name)
        if session_cfg is None:
            continue
            
        # Parse timezone-aware session start/end times
        orb_date = datetime.strptime(orb.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
        session_start = datetime(
            orb_date.year, orb_date.month, orb_date.day,
            session_cfg.start_hour, session_cfg.start_minute,
            tzinfo=timezone.utc
        )
        session_end = datetime(
            orb_date.year, orb_date.month, orb_date.day,
            session_cfg.end_hour, session_cfg.end_minute,
            tzinfo=timezone.utc
        )
        if session_end <= session_start:
            session_end += timedelta(days=1)
            
        # Determine precise status and corresponding color code
        if latest_bar_time < session_start:
            status_label = "AWAITING SESSION"
            status_color = "#718096"
            status_bg = "rgba(113, 128, 150, 0.1)"
        elif session_start <= latest_bar_time < orb.range_close_time:
            status_label = "FORMING RANGE"
            status_color = "#ff9800"
            status_bg = "rgba(255, 152, 0, 0.1)"
        else:
            if orb.breakout_direction == "bullish":
                status_label = "BULLISH BREAKOUT"
                status_color = "#26a69a"
                status_bg = "rgba(38, 166, 154, 0.1)"
            elif orb.breakout_direction == "bearish":
                status_label = "BEARISH BREAKOUT"
                status_color = "#ef5350"
                status_bg = "rgba(239, 83, 80, 0.1)"
            else:
                if latest_bar_time < session_end:
                    status_label = "ACTIVE / NO BREAKOUT"
                    status_color = "#00bcd4"
                    status_bg = "rgba(0, 188, 212, 0.1)"
                else:
                    status_label = "CLOSED / NO BREAKOUT"
                    status_color = "#4a5568"
                    status_bg = "rgba(74, 85, 104, 0.1)"
        
        card_html = f"""
        <div style="background-color: #1a202c; border-radius: 12px; padding: 22px; margin-bottom: 20px; border-left: 6px solid {status_color}; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <span style="font-size: 1.3rem; font-weight: 700; color: #fff; font-family: 'Outfit', sans-serif;">{orb.session_name} Session</span>
                <span style="font-size: 0.85rem; font-weight: 700; color: {status_color}; background-color: {status_bg}; padding: 4px 12px; border-radius: 20px; border: 1px solid {status_color}33; font-family: 'Outfit', sans-serif;">
                    {status_label}
                </span>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; font-size: 0.9rem; color: #cbd5e0; font-family: 'Outfit', sans-serif;">
                <div>
                    <span style="color: #718096; font-weight: 600; display: block; text-transform: uppercase; font-size: 0.75rem;">Opening Range</span>
                    <b>High:</b> {f"{orb.range_high:.2f}" if orb.range_high is not None else "N/A"}<br/>
                    <b>Low:</b> {f"{orb.range_low:.2f}" if orb.range_low is not None else "N/A"}<br/>
                    <b>Bars in Range:</b> {orb.bars_in_range}
                </div>
                <div>
                    <span style="color: #718096; font-weight: 600; display: block; text-transform: uppercase; font-size: 0.75rem;">Range Window</span>
                    <b>Open:</b> {orb.range_open_time.strftime('%H:%M') if orb.range_open_time else "N/A"} UTC<br/>
                    <b>Close:</b> {orb.range_close_time.strftime('%H:%M') if orb.range_close_time else "N/A"} UTC
                </div>
                <div>
                    <span style="color: #718096; font-weight: 600; display: block; text-transform: uppercase; font-size: 0.75rem;">Breakout Details</span>
                    <b>Price:</b> {f"{orb.breakout_price:.2f}" if orb.breakout_price is not None else "N/A"}<br/>
                    <b>Time:</b> {orb.breakout_time.strftime('%H:%M') if orb.breakout_time else "N/A"} UTC
                </div>
                <div>
                    <span style="color: #718096; font-weight: 600; display: block; text-transform: uppercase; font-size: 0.75rem;">Retest Confirmation</span>
                    <b>Confirmed:</b> {"Yes ✔️" if orb.retest_confirmed else "No ❌"}<br/>
                    <b>Retest Time:</b> {orb.retest_time.strftime('%H:%M') if orb.retest_time else "N/A"} UTC
                </div>
            </div>
        </div>
        """
        clean_html = "".join([line.strip() for line in card_html.split("\n")])
        st.markdown(clean_html, unsafe_allow_html=True)



def render_fvg_analysis(df: pd.DataFrame, smc_result: dict, config: dict):
    """Render the dedicated Fair Value Gaps analytics and grid page."""
    st.markdown("## 🔍 Fair Value Gaps & Imbalance Analytics")
    
    fvgs = smc_result.get("fvgs", [])
    
    if not fvgs:
        st.info("No FVGs detected in the current buffer range.")
        return
        
    # Stats
    total_fvgs = len(fvgs)
    mitigated = sum(1 for f in fvgs if f["mitigated"])
    unmitigated = total_fvgs - mitigated
    bullish = sum(1 for f in fvgs if f["direction"] == "bullish")
    bearish = sum(1 for f in fvgs if f["direction"] == "bearish")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total FVGs", total_fvgs)
    with col2:
        st.metric("Active (Unmitigated)", unmitigated)
    with col3:
        st.metric("Mitigated", mitigated)
    with col4:
        ratio = bullish / bearish if bearish > 0 else bullish
        st.metric("Bullish/Bearish Ratio", f"{ratio:.1f}")
        
    st.divider()
    
    # Filter controls
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        f_status = st.multiselect("Filter by Status", ["Active", "Mitigated"], default=["Active", "Mitigated"])
    with f_col2:
        f_dir = st.multiselect("Filter by Direction", ["Bullish", "Bearish"], default=["Bullish", "Bearish"])
        
    # Convert to list for DataFrame
    fvg_records = []
    for f in fvgs:
        status_str = "Active" if not f["mitigated"] else "Mitigated"
        dir_str = "Bullish" if f["direction"] == "bullish" else "Bearish"
        
        if status_str not in f_status or dir_str not in f_dir:
            continue
            
        fvg_records.append({
            "Time": f["time"],
            "Direction": dir_str,
            "Top": f["top"],
            "Bottom": f["bottom"],
            "Gap (Pips)": f["gap_pips"],
            "Status": status_str,
            "Mitigation Time": f["mitigation_time"]
        })
        
    if not fvg_records:
        st.info("No FVGs match the current filter selection.")
        return
        
    fvg_df = pd.DataFrame(fvg_records)
    fvg_df.sort_values(by="Time", ascending=False, inplace=True)
    
    # Render interactive DataFrame
    st.write("### 📋 Fair Value Gaps Log")
    st.dataframe(
        fvg_df,
        column_config={
            "Time": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "Mitigation Time": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "Gap (Pips)": st.column_config.NumberColumn(format="%.1f"),
            "Top": st.column_config.NumberColumn(format="%.2f"),
            "Bottom": st.column_config.NumberColumn(format="%.2f"),
        },
        use_container_width=True
    )
    
    # Plotly Chart: FVG size distribution
    st.divider()
    st.write("### 📊 FVG Size (Pips) Distribution")
    
    import plotly.express as px
    fig = px.histogram(
        fvg_df, 
        x="Gap (Pips)", 
        color="Direction",
        nbins=20,
        color_discrete_map={"Bullish": "#26a69a", "Bearish": "#ef5350"},
        template="plotly_dark",
        title="Distribution of Fair Value Gap Sizes"
    )
    fig.update_layout(
        yaxis_title="Count",
        xaxis_title="Imbalance Size (Pips)",
        bargap=0.05
    )
    st.plotly_chart(fig, use_container_width=True)


def render_signals_page(signals: list[Signal], smc_result: dict, config: dict):
    """Render the scored confluence signals cards and triggers metrics."""
    st.markdown("## 🔔 Scored Confluence Signals")
    
    if not signals:
        st.info("No active confluence signals detected for the current range/date.")
        return
        
    # Top Metrics
    total_signals = len(signals)
    bullish = sum(1 for s in signals if s.direction == "bullish")
    bearish = sum(1 for s in signals if s.direction == "bearish")
    avg_score = sum(s.total_score for s in signals) / total_signals if total_signals > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Signals", total_signals)
    with col2:
        st.metric("Bullish vs Bearish", f"{bullish} 🟢 / {bearish} 🔴")
    with col3:
        st.metric("Average Score", f"{avg_score:.2f}")
        
    st.divider()

    # ── High-Impact USD News Calendar ──
    news_enabled = config.get("signals", {}).get("news_filter_enabled", True)
    news_events = st.session_state.get("news_events", [])
    
    if news_enabled:
        with st.expander("📅 High-Impact USD News Calendar (This Week)", expanded=False):
            if news_events:
                news_html = """
                <table style="width: 100%; border-collapse: collapse; font-family: 'Outfit', sans-serif; font-size: 0.9rem;">
                    <thead>
                        <tr style="border-bottom: 2px solid #2d3748; text-align: left; color: #a0aec0;">
                            <th style="padding: 8px;">Event Title</th>
                            <th style="padding: 8px;">Release Time (UTC)</th>
                            <th style="padding: 8px;">Impact</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                for event in news_events:
                    news_html += f"""
                        <tr style="border-bottom: 1px solid #2d3748; color: #e2e8f0;">
                            <td style="padding: 8px; font-weight: 600;">{event['title']}</td>
                            <td style="padding: 8px;">{event['time'].strftime('%Y-%m-%d %H:%M')}</td>
                            <td style="padding: 8px;"><span style="background-color: rgba(239, 83, 80, 0.2); color: #ef5350; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">HIGH</span></td>
                        </tr>
                    """
                news_html += "</tbody></table>"
                st.markdown("".join([line.strip() for line in news_html.split("\n")]), unsafe_allow_html=True)
            else:
                st.info("No high-impact USD news releases scheduled for this week (or calendar offline).")
        st.divider()
    
    # Signal Breakdown Cards
    st.write("### 🚨 Active Signals Breakdown")
    for sig in signals:
        direction_class = "signal-card-bullish" if sig.direction == "bullish" else "signal-card-bearish"
        direction_icon = "🟢" if sig.direction == "bullish" else "🔴"
        
        triggered_badges = "".join(
            f'<span class="rule-badge rule-badge-triggered">✔️ {rule}</span>'
            for rule in sig.rules_triggered
        )
        missed_badges = "".join(
            f'<span class="rule-badge rule-badge-missed">❌ {rule}</span>'
            for rule in sig.rules_missed
        )
        
        score_pct = sig.score_pct
        score_color = "#26a69a" if sig.direction == "bullish" else "#ef5350"
        time_str = sig.timestamp.strftime('%Y-%m-%d %H:%M') if sig.timestamp else 'N/A'
        
        card_html = f"""
        <div class="signal-card {direction_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #fff;">
                    {direction_icon} {sig.session_name} Session Breakout
                </span>
                <span style="font-size: 1rem; font-weight: 600; color: {score_color}; background-color: rgba(255,255,255,0.05); padding: 4px 10px; border-radius: 20px;">
                    Score: {sig.total_score:.1f} / {sig.max_possible_score:.1f} ({sig.score_pct:.0%})
                </span>
            </div>
            <div style="font-size: 0.9rem; color: #a0aec0; margin-bottom: 12px; display: flex; gap: 20px;">
                <span>Signal Time: <b>{time_str} UTC</b></span>
                <span>Entry: <b>{sig.details.get('entry_price', 'N/A')}</b></span>
            </div>
            
            <div style="display: flex; gap: 15px; margin-bottom: 15px; background: rgba(0,0,0,0.25); padding: 10px 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="flex: 1;">
                    <span style="display: block; font-size: 0.7rem; color: #a0aec0; text-transform: uppercase; font-weight: 600; margin-bottom: 3px;">Stop Loss (SL)</span>
                    <b style="font-size: 1.15rem; color: #f28b82;">{sig.sl}</b>
                </div>
                <div style="flex: 1;">
                    <span style="display: block; font-size: 0.7rem; color: #a0aec0; text-transform: uppercase; font-weight: 600; margin-bottom: 3px;">Take Profit (TP)</span>
                    <b style="font-size: 1.15rem; color: #81c784;">{sig.tp}</b>
                </div>
                <div style="flex: 1;">
                    <span style="display: block; font-size: 0.7rem; color: #a0aec0; text-transform: uppercase; font-weight: 600; margin-bottom: 3px;">R:R Ratio</span>
                    <b style="font-size: 1.15rem; color: #64b5f6;">{sig.rr_ratio}:1</b>
                </div>
            </div>
            
            <div style="margin-bottom: 15px;">
                <div style="background-color: rgba(255, 255, 255, 0.1); height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background-color: {score_color}; width: {score_pct * 100}%; height: 100%;"></div>
                </div>
            </div>
            <div>
                <div style="font-size: 0.85rem; font-weight: 600; color: #cbd5e0; margin-bottom: 6px;">Triggered Rules:</div>
                <div style="margin-bottom: 10px;">{triggered_badges or 'None'}</div>
                <div style="font-size: 0.85rem; font-weight: 600; color: #cbd5e0; margin-bottom: 6px;">Missed Rules:</div>
                <div>{missed_badges or 'None'}</div>
            </div>
        </div>
        """
        clean_html = "".join([line.strip() for line in card_html.split("\n")])
        st.markdown(clean_html, unsafe_allow_html=True)
        
    # Rules Frequency breakdown bar chart
    st.divider()
    st.write("### 📊 Confluence Rules Trigger Frequency")
    
    rule_freq = {}
    for sig in signals:
        for rule in sig.rules_triggered:
            rule_freq[rule] = rule_freq.get(rule, 0) + 1
            
    if rule_freq:
        import plotly.express as px
        rf_df = pd.DataFrame([{"Rule": r, "Triggers": count} for r, count in rule_freq.items()])
        rf_df.sort_values(by="Triggers", ascending=True, inplace=True)
        
        fig = px.bar(
            rf_df, 
            y="Rule", 
            x="Triggers", 
            orientation="h",
            template="plotly_dark",
            color="Triggers",
            color_continuous_scale="Viridis",
            title="Frequency of Triggered Confluence Factors"
        )
        fig.update_layout(
            xaxis_title="Number of Triggers",
            yaxis_title="Rule Name"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No statistics available for triggered rules.")


# ──────────────────────────────────────────────────────────────
# Main Dashboard Router
# ──────────────────────────────────────────────────────────────

def main():
    config_path = get_config_path()

    try:
        feed, config = init_data_feed(config_path)
    except Exception as e:
        st.error(f"Failed to initialize MT5 data feed: {e}")
        st.info(
            "Make sure MetaTrader 5 is running and logged in. "
            "Check config.yaml for correct symbol name."
        )
        return

    # Inject Custom CSS Styling
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Premium metric values styling */
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 0.8rem;
            font-weight: 600;
            color: #8892b0;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Premium custom signal cards */
        .signal-card {
            background-color: #1a202c;
            border-radius: 12px;
            padding: 22px;
            margin-bottom: 18px;
            border-left: 6px solid #a0aec0;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .signal-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.35);
        }
        
        .signal-card-bullish {
            border-left-color: #26a69a !important;
            background: linear-gradient(135deg, #1a202c 0%, rgba(38, 166, 154, 0.08) 100%);
        }
        
        .signal-card-bearish {
            border-left-color: #ef5350 !important;
            background: linear-gradient(135deg, #1a202c 0%, rgba(239, 83, 80, 0.08) 100%);
        }
        
        .rule-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            margin: 3px;
        }
        
        .rule-badge-triggered {
            background-color: rgba(38, 166, 154, 0.15);
            color: #4db6ac;
            border: 1px solid rgba(38, 166, 154, 0.3);
        }
        
        .rule-badge-missed {
            background-color: rgba(239, 83, 80, 0.15);
            color: #e57373;
            border: 1px solid rgba(239, 83, 80, 0.3);
        }
        
        /* Hide loading running indicator */
        [data-testid="stStatusWidget"] {
            visibility: hidden;
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ── Sidebar Controls ──
    with st.sidebar:
        st.title("⚙️ Controls")

        from data_feed import MT5_AVAILABLE
        if not MT5_AVAILABLE:
            st.info("💡 Operating in **Live Demo Mode** (MT5 connection is simulated).")

        # Sidebar navigation
        st.subheader("Navigation")
        page = st.radio("Go to", ["📈 Main Dashboard", "🔍 FVG Analysis", "🔔 Trading Signals", "📋 Session Details"])
        st.divider()

        mode = st.radio("Mode", ["Live", "Historical"], index=0)

        selected_date = datetime.now(timezone.utc)
        if mode == "Historical":
            date_pick = st.date_input("Select Date", selected_date.date())
            selected_date = datetime.combine(date_pick, datetime.min.time(), tzinfo=timezone.utc)

        chart_bars = st.slider(
            "Chart Bars",
            min_value=100, max_value=2000,
            value=config.get("dashboard", {}).get("chart_bars", 500),
            step=100
        )

        refresh_min = st.slider(
            "Refresh Interval (min)",
            min_value=1, max_value=60,
            value=config.get("dashboard", {}).get("refresh_minutes", 30),
            key="refresh_interval"
        )

        st.divider()
        st.subheader("Session Toggles")
        sessions_cfg = config.get("sessions", {})
        show_sessions = {}
        for key, sess in sessions_cfg.items():
            show_sessions[sess.get("name", key)] = st.checkbox(
                sess.get("name", key), value=True
            )

        st.divider()
        st.subheader("Overlay Toggles")
        show_obs = st.checkbox("Order Blocks", value=True)
        show_fvgs = st.checkbox("Fair Value Gaps", value=True)
        show_swings = st.checkbox("Swing Points", value=True)
        show_bos = st.checkbox("BOS / CHoCH", value=True)

        st.divider()
        st.subheader("SMC Parameters")
        swing_lookback = st.slider(
            "Swing Lookback", 2, 15,
            value=config.get("smc", {}).get("swing_lookback", 5)
        )
        min_score = st.slider(
            "Min Signal Score", 0.0, 7.0,
            value=config.get("signals", {}).get("min_score", 3.0),
            step=0.5
        )

    # ── Fetch Data ──
    smc_config = config.get("smc", {}).copy()
    smc_config["swing_lookback"] = swing_lookback

    if mode == "Live":
        # Ensure polling is running
        if not (feed._poll_thread and feed._poll_thread.is_alive()):
            feed.start_live_polling()

        df = feed.get_buffer()
    else:
        # Historical: fetch data around selected date
        start = selected_date - timedelta(hours=6)
        end = selected_date + timedelta(hours=24)
        try:
            df = feed.fetch_historical_bars(start, end)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.info("No data available for this range. Markets are closed on weekends.")
            return

    if df.empty:
        st.warning("No data available. Waiting for market data...")
        if mode == "Live":
            time.sleep(refresh_sec)
            st.rerun()
        return

    # ── Run SMC/ORB Analysis ──
    sessions = load_sessions(config)
    # Filter to only enabled sessions
    sessions = [s for s in sessions if show_sessions.get(s.name, True)]

    smc_result = analyze_structure(df, smc_config)

    # Filter overlays based on toggles
    if not show_obs:
        smc_result["order_blocks"] = []
    if not show_fvgs:
        smc_result["fvgs"] = []
    if not show_swings:
        smc_result["swing_df"] = None
    if not show_bos:
        smc_result["bos"] = []
        smc_result["choch"] = []

    pip_size = smc_config.get("pip_size", 0.01)
    orb_results = analyze_all_sessions(df, sessions, selected_date, pip_size=pip_size)

    rules = load_rules(config)
    
    # Fetch ADR and USD News Events
    adr = feed.fetch_adr(days=5)
    from utils import fetch_high_impact_news
    if "news_events" not in st.session_state:
        st.session_state.news_events = fetch_high_impact_news()
        
    signals = generate_signals(
        orb_results,
        smc_result,
        rules,
        min_score,
        config=config,
        adr=adr,
        news_events=st.session_state.news_events
    )

    # ── Route Pages ──
    if page == "📈 Main Dashboard":
        render_main_dashboard(df, orb_results, smc_result, signals, chart_bars)
    elif page == "🔍 FVG Analysis":
        render_fvg_analysis(df, smc_result, config)
    elif page == "🔔 Trading Signals":
        render_signals_page(signals, smc_result, config)
    elif page == "📋 Session Details":
        render_session_details(orb_results, df.index[-1], sessions, adr=adr)

    # ── Auto-Refresh (Live Mode Only) ──
    if mode == "Live":
        time.sleep(refresh_min * 60)
        st.rerun()


if __name__ == "__main__":
    main()
