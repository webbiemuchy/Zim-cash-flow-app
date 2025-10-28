import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backend import ZimbabweCashFlowModel

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Zimbabwe Cash Flow Dashboard — Enhanced",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Enhanced Styles
# ----------------------------
ENHANCED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
    --bg-gradient-start:#0f172a;
    --bg-gradient-end:#1e293b;
    --card-bg:rgba(255,255,255,0.95);
    --card-glass:rgba(255,255,255,0.1);
    --muted:#64748b;
    --accent:#0f172a;
    --success:#10b981;
    --danger:#ef4444;
    --warning:#f59e0b;
    --info:#3b82f6;
    --accent-2:#06b6d4;
    --purple:#a855f7;
}

* { font-family: 'Inter', sans-serif; }

body { 
    background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
}

.main { padding: 1rem; }

/* Header Section */
.dashboard-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    position: relative;
    overflow: hidden;
}

.dashboard-header::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.3) 0%, transparent 70%);
    border-radius: 50%;
}

.header-content {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}

.header-title {
    color: white;
    font-size: 28px;
    font-weight: 800;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.header-subtitle {
    color: rgba(255,255,255,0.8);
    font-size: 14px;
    margin: 0.5rem 0 0 0;
}

/* Enhanced KPI Cards */
.kpi-card {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    border: 1px solid rgba(15,23,42,0.05);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--accent-2), var(--info));
}

.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.12);
}

.kpi-label {
    color: var(--muted);
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.kpi-value {
    font-size: 32px;
    font-weight: 800;
    color: var(--accent);
    margin: 0.5rem 0;
    line-height: 1.2;
}

.kpi-change {
    font-size: 14px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
}

.kpi-change.positive {
    color: var(--success);
    background: rgba(16,185,129,0.1);
}

.kpi-change.negative {
    color: var(--danger);
    background: rgba(239,68,68,0.1);
}

.kpi-change.neutral {
    color: var(--muted);
    background: rgba(100,116,139,0.1);
}

/* Risk Badges */
.risk-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-weight: 700;
    font-size: 16px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.risk-low { 
    background: rgba(16,185,129,0.15);
    color: #059669;
    border: 2px solid #10b981;
}
.risk-moderate { 
    background: rgba(245,158,11,0.15);
    color: #d97706;
    border: 2px solid #f59e0b;
}
.risk-high { 
    background: rgba(239,68,68,0.15);
    color: #dc2626;
    border: 2px solid #ef4444;
}
.risk-critical { 
    background: rgba(153,27,27,0.15);
    color: #991b1b;
    border: 2px solid #b91c1c;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

/* Quick Actions */
.quick-action-btn {
    background: white;
    border: 2px solid var(--info);
    color: var(--info);
    padding: 0.75rem 1.5rem;
    border-radius: 10px;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.quick-action-btn:hover {
    background: var(--info);
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(59,130,246,0.3);
}

/* Info Box */
.info-box {
    background: rgba(59,130,246,0.1);
    border-left: 4px solid var(--info);
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.warning-box {
    background: rgba(245,158,11,0.1);
    border-left: 4px solid var(--warning);
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.success-box {
    background: rgba(16,185,129,0.1);
    border-left: 4px solid var(--success);
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

/* Section Headers */
.section-header {
    font-size: 20px;
    font-weight: 700;
    color: var(--accent);
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid var(--accent-2);
}

/* Metric Grid */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-item {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.metric-item-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
}

.metric-item-value {
    font-size: 20px;
    font-weight: 700;
    color: var(--accent);
    margin-top: 0.25rem;
}

/* Footer */
.dashboard-footer {
    text-align: center;
    padding: 2rem;
    color: var(--muted);
    font-size: 12px;
    margin-top: 3rem;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-fade-in {
    animation: fadeIn 0.5s ease-out;
}

/* Responsive */
@media (max-width: 768px) {
    .header-title { font-size: 20px; }
    .kpi-value { font-size: 24px; }
    .metric-grid { grid-template-columns: 1fr; }
}
</style>
"""

PRESENTATION_MODE_CSS = """
<style>
.header-title { font-size: 36px !important; }
.kpi-value { font-size: 40px !important; }
.section-header { font-size: 24px !important; }
</style>
"""

st.markdown(ENHANCED_CSS, unsafe_allow_html=True)

# ----------------------------
# Utilities
# ----------------------------
def to_pydatetime_safe(x):
    try:
        if isinstance(x, pd.Timestamp):
            return x.to_pydatetime()
        if isinstance(x, dt.datetime):
            return x
        return pd.to_datetime(x).to_pydatetime()
    except Exception:
        return str(x)

def df_to_csv_bytes(df: pd.DataFrame):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode('utf-8')

def fig_to_png_bytes(fig):
    try:
        img_bytes = fig.to_image(format="png", engine="kaleido")
        return img_bytes
    except Exception:
        try:
            img_bytes = fig.to_image(format="png")
            return img_bytes
        except Exception:
            return None

def calculate_trend(current, previous):
    """Calculate percentage change and return formatted string with arrow"""
    if previous == 0:
        return "N/A", "neutral"
    
    change = ((current - previous) / abs(previous)) * 100
    
    if abs(change) < 0.1:
        arrow = "→"
        direction = "neutral"
    elif change > 0:
        arrow = "↑"
        direction = "positive"
    else:
        arrow = "↓"
        direction = "negative"
    
    return f"{arrow} {abs(change):.1f}%", direction

def generate_summary_stats(net_flows, forecast_df):
    """Generate comprehensive summary statistics"""
    stats = {
        'Total Simulation Days': len(net_flows),
        'Average Daily Inflow': f"${net_flows['inflow'].mean():,.2f}",
        'Average Daily Outflow': f"${net_flows['outflow'].mean():,.2f}",
        'Average Daily Net': f"${net_flows['net'].mean():,.2f}",
        'Max Daily Net': f"${net_flows['net'].max():,.2f}",
        'Min Daily Net': f"${net_flows['net'].min():,.2f}",
        'Net Flow Volatility': f"${net_flows['net'].std():,.2f}",
        'Peak Balance': f"${net_flows['cumulative'].max():,.2f}",
        'Lowest Balance': f"${net_flows['cumulative'].min():,.2f}",
    }
    
    if not forecast_df.empty:
        stats['Forecast Days'] = len(forecast_df)
        stats['Predicted End Balance'] = f"${forecast_df['cumulative_forecast'].iloc[-1]:,.2f}"
    
    return stats

# ----------------------------
# Enhanced Header
# ----------------------------
st.markdown("""
<div class='dashboard-header'>
    <div class='header-content'>
        <img src='logo.svg' width='60' style='border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);'>
        <div>
            <div class='header-title'>Zimbabwe Cash Flow Intelligence Platform</div>
            <div class='header-subtitle'>Advanced Multi-Currency Simulation • AI-Powered Risk Analysis • ARIMA Forecasting</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Enhanced Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Simulation Configuration")
    
    with st.form("controls"):
        st.markdown("#### 💰 Financial Parameters")
        initial_balance = st.number_input(
            "Initial USD Balance", 
            value=25000, 
            min_value=0, 
            step=500, 
            format="%d",
            help="Starting cash balance in USD"
        )
        
        st.markdown("#### 📅 Time Parameters")
        col1, col2 = st.columns(2)
        with col1:
            sim_days = st.number_input(
                "Historical Days", 
                min_value=30, 
                max_value=365, 
                value=180, 
                step=30,
                help="Days for ML training"
            )
        with col2:
            forecast_days = st.number_input(
                "Forecast Days", 
                min_value=15, 
                max_value=90, 
                value=30, 
                step=15,
                help="AI forecast horizon"
            )
        
        st.markdown("#### 📊 Business Parameters")
        business_scale = st.slider(
            "Business Scale Multiplier", 
            min_value=0.5, 
            max_value=3.0, 
            value=1.0, 
            step=0.1,
            help="Scale transaction volumes"
        )
        
        st.markdown("#### 💱 Currency Volatility")
        col3, col4 = st.columns(2)
        with col3:
            zig_volatility = st.slider(
                "ZiG Volatility", 
                min_value=0.005, 
                max_value=0.15, 
                value=0.02, 
                step=0.001, 
                format="%.3f"
            )
        with col4:
            zar_volatility = st.slider(
                "ZAR Volatility", 
                min_value=0.002, 
                max_value=0.07, 
                value=0.01, 
                step=0.001, 
                format="%.3f"
            )

        with st.expander("🔬 Advanced Analytics", expanded=False):
            seed_option = st.checkbox("Fixed Random Seed", value=False)
            seed_value = st.number_input("Seed Value", value=42, step=1) if seed_option else None
            car_confidence = st.slider("CaR Confidence Level", min_value=0.90, max_value=0.999, value=0.95, step=0.005)
            arima_order_text = st.text_input("ARIMA Order (p,d,q)", value="5,1,0")

        st.markdown("---")
        presentation_mode = st.checkbox("🎯 Presentation Mode", value=False)
        
        col_run, col_reset = st.columns(2)
        with col_run:
            run = st.form_submit_button("▶️ Run", use_container_width=True)
        with col_reset:
            reset = st.form_submit_button("🔄 Reset", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📚 Quick Guide")
    st.markdown("""
    **Getting Started:**
    1. Configure parameters above
    2. Click Run to analyze
    3. Explore visualizations
    4. Export results as needed
    
    **Key Features:**
    - 💰 Real-time cash flow tracking
    - 🎯 AI-powered forecasting
    - 📊 Multi-currency analysis
    - ⚠️ Risk assessment
    """)

# Reset handling
if reset:
    for k in ['run_model','net_flows','forecast_df','category_analysis','inflows','outflows','params']:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# Persist params
st.session_state['params'] = {
    'initial_balance': initial_balance,
    'sim_days': sim_days,
    'forecast_days': forecast_days,
    'business_scale': business_scale,
    'zig_volatility': zig_volatility,
    'zar_volatility': zar_volatility,
    'seed_option': seed_option,
    'seed_value': int(seed_value) if seed_option else None,
    'car_confidence': car_confidence,
    'arima_order_text': arima_order_text
}

if presentation_mode:
    st.markdown(PRESENTATION_MODE_CSS, unsafe_allow_html=True)

if run:
    st.session_state['run_model'] = True

# ----------------------------
# Execute Model
# ----------------------------
if st.session_state.get('run_model', False):
    params = st.session_state['params']

    if params.get('seed_option') and params.get('seed_value') is not None:
        np.random.seed(int(params['seed_value']))

    try:
        p, d, q = [int(x.strip()) for x in params.get('arima_order_text', "5,1,0").split(",")]
        arima_order = (p, d, q)
    except Exception:
        arima_order = (5, 1, 0)
        st.warning("Invalid ARIMA order format. Using default (5,1,0)")

    model = ZimbabweCashFlowModel(
        initial_balance=float(params['initial_balance']),
        zig_volatility=float(params['zig_volatility']),
        zar_volatility=float(params['zar_volatility']),
        business_scale=float(params['business_scale'])
    )

    with st.spinner("🔄 Running simulation and AI analysis..."):
        progress_bar = st.progress(0)
        
        progress_bar.progress(20)
        inflows, outflows = model.simulate_transactions(days=int(params['sim_days']))
        
        progress_bar.progress(40)
        net_flows = model.calculate_net_flows(inflows, outflows)

        if net_flows.empty:
            st.error("⚠️ Simulation produced no cash flows. Please adjust parameters and retry.")
            st.session_state['run_model'] = False
            st.stop()
        
        progress_bar.progress(60)
        net_flows = model.analyze_liquidity(net_flows)
        net_flows_with_car = model.cash_at_risk_analysis(net_flows, confidence_level=float(params.get('car_confidence', 0.95)))
        
        progress_bar.progress(80)
        forecast_df = model.ml_forecast_net_flow(net_flows_with_car, forecast_days=int(params['forecast_days']), order=arima_order)
        category_analysis = model.categorized_analysis(inflows, outflows)

        net_flows_with_car['date'] = pd.to_datetime(net_flows_with_car['date'])
        if not forecast_df.empty:
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])

        st.session_state['net_flows'] = net_flows_with_car
        st.session_state['forecast_df'] = forecast_df
        st.session_state['category_analysis'] = category_analysis
        st.session_state['inflows'] = inflows
        st.session_state['outflows'] = outflows
        
        progress_bar.progress(100)
        st.success("✅ Analysis complete!")

# ----------------------------
# Results Display
# ----------------------------
if 'net_flows' not in st.session_state or st.session_state['net_flows'].empty:
    st.markdown("""
    <div class='info-box animate-fade-in'>
        <h3 style='margin:0 0 0.5rem 0; color: var(--info);'>👋 Welcome to the Dashboard</h3>
        <p style='margin:0; color: var(--muted);'>
            Configure your simulation parameters in the sidebar and click <strong>Run</strong> to begin analysis.
            The platform will generate comprehensive cash flow insights, risk assessments, and AI-powered forecasts.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 💰 Cash Flow Analysis")
        st.markdown("Track cumulative balance, daily flows, and liquidity metrics in real-time.")
    with col2:
        st.markdown("### 🧠 AI Forecasting")
        st.markdown("ARIMA-based predictions with confidence intervals for future cash positions.")
    with col3:
        st.markdown("### 📊 Risk Assessment")
        st.markdown("Cash-at-Risk analysis and multi-level liquidity risk indicators.")
    
    st.stop()

# Load results
net_flows = st.session_state['net_flows']
forecast_df = st.session_state.get('forecast_df', pd.DataFrame())
category_analysis = st.session_state.get('category_analysis', {})
inflows = st.session_state.get('inflows', pd.DataFrame())
outflows = st.session_state.get('outflows', pd.DataFrame())
params = st.session_state['params']

# ----------------------------
# Enhanced KPI Cards
# ----------------------------
st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

with k1:
    current_balance = float(net_flows['cumulative'].iloc[-1])
    prev_balance = float(net_flows['cumulative'].iloc[-2]) if len(net_flows) > 1 else current_balance
    trend_text, trend_dir = calculate_trend(current_balance, prev_balance)
    
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>💵 Current Balance</div>
        <div class='kpi-value'>${current_balance:,.2f}</div>
        <div class='kpi-change {trend_dir}'>{trend_text}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    avg_net = float(net_flows['net'].mean())
    recent_avg = float(net_flows['net'].tail(7).mean())
    trend_text, trend_dir = calculate_trend(recent_avg, avg_net)
    
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>📊 Avg Daily Net</div>
        <div class='kpi-value'>${avg_net:,.2f}</div>
        <div class='kpi-change {trend_dir}'>7-day: {trend_text}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    last_risk = str(net_flows['liquidity_risk'].iloc[-1])
    risk_class_map = {"Low":"risk-low","Moderate":"risk-moderate","High":"risk-high","Critical":"risk-critical"}
    risk_class = risk_class_map.get(last_risk, "risk-moderate")
    
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>⚠️ Liquidity Risk</div>
        <div class='risk-badge {risk_class}'>{last_risk}</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    worst_case = float(net_flows['worst_case_cumulative'].iloc[-1]) if 'worst_case_cumulative' in net_flows.columns else current_balance
    car_impact = current_balance - worst_case
    
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>🎯 Cash-at-Risk</div>
        <div class='kpi-value' style='color:#dc2626'>${worst_case:,.2f}</div>
        <div class='kpi-change negative'>Impact: ${car_impact:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Quick Actions
# ----------------------------
st.markdown("<div style='margin: 2rem 0;'>", unsafe_allow_html=True)
qa1, qa2, qa3, qa4 = st.columns(4)

with qa1:
    summary_stats = generate_summary_stats(net_flows, forecast_df)
    summary_text = "\n".join([f"{k}: {v}" for k, v in summary_stats.items()])
    st.download_button(
        "📄 Download Summary",
        data=summary_text,
        file_name="cash_flow_summary.txt",
        mime="text/plain",
        use_container_width=True
    )

with qa2:
    st.download_button(
        "💾 Export Net Flows",
        data=df_to_csv_bytes(net_flows),
        file_name="net_flows_detailed.csv",
        mime="text/csv",
        use_container_width=True
    )

with qa3:
    if not forecast_df.empty:
        st.download_button(
            "🔮 Export Forecast",
            data=df_to_csv_bytes(forecast_df),
            file_name="ai_forecast.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.button("🔮 No Forecast", disabled=True, use_container_width=True)

with qa4:
    if not inflows.empty and not outflows.empty:
        combined = pd.concat([
            inflows.assign(type='inflow'),
            outflows.assign(type='outflow')
        ])
        st.download_button(
            "📊 Export Transactions",
            data=df_to_csv_bytes(combined),
            file_name="all_transactions.csv",
            mime="text/csv",
            use_container_width=True
        )

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Main Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Cash Flow Analysis", "🧠 AI Forecasting", "📊 Category Breakdown", "📋 Detailed Data"])

# TAB 1: Enhanced Cash Flow
with tab1:
    st.markdown("<div class='section-header'>Cumulative Balance & Risk Timeline</div>", unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Main balance line
    fig.add_trace(go.Scatter(
        x=net_flows['date'],
        y=net_flows['cumulative'],
        mode='lines',
        name='Actual Balance',
        line=dict(color='#0f172a', width=3),
        fill='tozeroy',
        fillcolor='rgba(15,23,42,0.1)'
    ))
    
    # Worst case line
    if 'worst_case_cumulative' in net_flows.columns:
        fig.add_trace(go.Scatter(
            x=net_flows['date'],
            y=net_flows['worst_case_cumulative'],
            mode='lines',
            name='Worst-Case (CaR)',
            line=dict(color='#ef4444', width=2, dash='dash')
        ))
    
    # Initial balance reference
    fig.add_hline(
        y=params['initial_balance'],
        line_dash="dot",
        line_color="#64748b",
        annotation_text="Initial",
        annotation_position="right"
    )
    
    # Danger zone
    if 'worst_case_cumulative' in net_flows.columns and (net_flows['worst_case_cumulative'] < 0).any():
        low_val = float(net_flows['worst_case_cumulative'].min())
        fig.add_hrect(
            y0=low_val,
            y1=0,
            fillcolor="rgba(239,68,68,0.1)",
            line_width=0,
            annotation_text="Danger Zone",
            annotation_position="top left"
        )
    
    fig.update_layout(
        title='Simulated Cash Flow with AI-Predicted Worst-Case Scenario',
        xaxis_title='Date',
        yaxis_title='Balance (USD)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='rgba(248,250,252,0.5)',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Export button
    png_bytes = fig_to_png_bytes(fig)
    if png_bytes:
        st.download_button("💾 Download Chart PNG", data=png_bytes, file_name="cash_flow_timeline.png", mime="image/png")
    
    # Additional Metrics Grid
    st.markdown("<div class='section-header'>Key Performance Indicators</div>", unsafe_allow_html=True)
    
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        total_inflow = net_flows['inflow'].sum()
        st.metric("Total Inflows", f"${total_inflow:,.0f}")
    
    with m2:
        total_outflow = net_flows['outflow'].sum()
        st.metric("Total Outflows", f"${total_outflow:,.0f}")
    
    with m3:
        net_change = current_balance - params['initial_balance']
        st.metric("Net Change", f"${net_change:,.0f}", delta=f"{(net_change/params['initial_balance']*100):.1f}%")
    
    with m4:
        days_covered = net_flows['days_covered'].iloc[-1]
        st.metric("Days Covered", f"{days_covered:.0f}")
    
    with m5:
        volatility = net_flows['net'].std()
        st.metric("Net Flow Volatility", f"${volatility:,.0f}")
    
    # Daily Net Flow Chart
    st.markdown("<div class='section-header'>Daily Net Flow Pattern</div>", unsafe_allow_html=True)
    
    fig_net = go.Figure()
    
    colors = ['#10b981' if x >= 0 else '#ef4444' for x in net_flows['net']]
    
    fig_net.add_trace(go.Bar(
        x=net_flows['date'],
        y=net_flows['net'],
        name='Daily Net Flow',
        marker_color=colors,
        hovertemplate='<b>Date:</b> %{x}<br><b>Net Flow:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    fig_net.add_hline(y=0, line_dash="dash", line_color="#64748b", line_width=1)
    
    fig_net.update_layout(
        title='Daily Net Cash Flow (Inflows - Outflows)',
        xaxis_title='Date',
        yaxis_title='Net Flow (USD)',
        hovermode='x unified',
        showlegend=False,
        plot_bgcolor='rgba(248,250,252,0.5)',
        height=400
    )
    
    st.plotly_chart(fig_net, use_container_width=True)
    
    # Risk Analysis Table
    st.markdown("<div class='section-header'>Recent Cash-at-Risk Analysis</div>", unsafe_allow_html=True)
    
    car_cols = ['date', 'cumulative']
    if 'cash_at_risk' in net_flows.columns: car_cols.append('cash_at_risk')
    if 'worst_case_cumulative' in net_flows.columns: car_cols.append('worst_case_cumulative')
    if 'worst_case_risk' in net_flows.columns: car_cols.append('worst_case_risk')
    if 'liquidity_risk' in net_flows.columns: car_cols.append('liquidity_risk')
    
    display_df = net_flows[car_cols].tail(30).copy()
    display_df['date'] = display_df['date'].dt.date
    
    st.dataframe(
        display_df.style.format({
            'cumulative': '${:,.2f}',
            'cash_at_risk': '${:,.2f}',
            'worst_case_cumulative': '${:,.2f}'
        }),
        use_container_width=True,
        height=400
    )

# TAB 2: Enhanced Forecasting
with tab2:
    st.markdown("<div class='section-header'>AI-Powered ARIMA Forecast</div>", unsafe_allow_html=True)
    
    if forecast_df is None or forecast_df.empty:
        st.markdown("""
        <div class='warning-box'>
            <strong>⚠️ Forecast Unavailable</strong><br>
            The ARIMA model could not generate a forecast. This may occur if:
            <ul>
                <li>Historical data is insufficient (< 10 days)</li>
                <li>The ARIMA order parameters are invalid</li>
                <li>The time series contains too many zeros or missing values</li>
            </ul>
            Try increasing the historical simulation days or adjusting ARIMA parameters.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Forecast metrics
        fm1, fm2, fm3, fm4 = st.columns(4)
        
        with fm1:
            forecast_end_balance = forecast_df['cumulative_forecast'].iloc[-1]
            st.metric("Predicted End Balance", f"${forecast_end_balance:,.0f}")
        
        with fm2:
            forecast_change = forecast_end_balance - current_balance
            st.metric("Forecast Change", f"${forecast_change:,.0f}", delta=f"{(forecast_change/current_balance*100):.1f}%")
        
        with fm3:
            avg_forecast_net = forecast_df['net_forecast'].mean()
            st.metric("Avg Forecast Net", f"${avg_forecast_net:,.0f}")
        
        with fm4:
            forecast_volatility = forecast_df['net_forecast'].std()
            st.metric("Forecast Volatility", f"${forecast_volatility:,.0f}")
        
        # Net Flow Forecast Chart
        st.markdown("<div class='section-header'>Daily Net Flow Forecast with Confidence Interval</div>", unsafe_allow_html=True)
        
        fig_fore = go.Figure()
        
        # Confidence interval
        fig_fore.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['upper_ci'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_fore.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['lower_ci'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(59,130,246,0.2)',
            line=dict(width=0),
            name='95% Confidence Interval'
        ))
        
        # Forecast line
        fig_fore.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['net_forecast'],
            mode='lines+markers',
            name='Forecast Mean',
            line=dict(color='#0f172a', width=3),
            marker=dict(size=6)
        ))
        
        fig_fore.update_layout(
            title='AI/ML Daily Net Flow Forecast (ARIMA)',
            xaxis_title='Date',
            yaxis_title='Net Flow (USD)',
            hovermode='x unified',
            plot_bgcolor='rgba(248,250,252,0.5)',
            height=500
        )
        
        st.plotly_chart(fig_fore, use_container_width=True)
        
        png_bytes_fore = fig_to_png_bytes(fig_fore)
        if png_bytes_fore:
            st.download_button("💾 Download Forecast PNG", data=png_bytes_fore, file_name="netflow_forecast.png", mime="image/png")
        
        # Cumulative Balance Forecast
        st.markdown("<div class='section-header'>Historical vs Forecast Cumulative Balance</div>", unsafe_allow_html=True)
        
        historical_data = net_flows[['date', 'cumulative']].rename(columns={'cumulative': 'cumulative_forecast'})
        combined_df = pd.concat([historical_data, forecast_df[['date', 'cumulative_forecast']]]).reset_index(drop=True)
        
        fig_cumulative = go.Figure()
        
        # Historical portion
        fig_cumulative.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['cumulative_forecast'],
            mode='lines',
            name='Historical Balance',
            line=dict(color='#0f172a', width=3)
        ))
        
        # Forecast portion
        fig_cumulative.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['cumulative_forecast'],
            mode='lines',
            name='Forecast Balance',
            line=dict(color='#10b981', width=3, dash='dash')
        ))
        
        # Forecast start marker
        last_hist_date = to_pydatetime_safe(historical_data['date'].iloc[-1])
        y_min = float(combined_df['cumulative_forecast'].min())
        y_max = float(combined_df['cumulative_forecast'].max())
        
        fig_cumulative.add_shape(
            type="line",
            x0=last_hist_date,
            x1=last_hist_date,
            y0=y_min,
            y1=y_max,
            line=dict(color="#f59e0b", dash="dash", width=2)
        )
        
        fig_cumulative.add_annotation(
            x=last_hist_date,
            y=y_max,
            text="← Historical | Forecast →",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            bgcolor="rgba(245,158,11,0.1)",
            bordercolor="#f59e0b",
            borderwidth=2
        )
        
        fig_cumulative.update_layout(
            title='Combined Historical & AI Forecast Cumulative Balance',
            xaxis_title='Date',
            yaxis_title='Cumulative Balance (USD)',
            hovermode='x unified',
            plot_bgcolor='rgba(248,250,252,0.5)',
            height=500
        )
        
        st.plotly_chart(fig_cumulative, use_container_width=True)
        
        png_bytes_cum = fig_to_png_bytes(fig_cumulative)
        if png_bytes_cum:
            st.download_button("💾 Download Cumulative PNG", data=png_bytes_cum, file_name="cumulative_forecast.png", mime="image/png")
        
        # Forecast Data Table
        with st.expander("📊 View Forecast Data Table", expanded=False):
            forecast_display = forecast_df.copy()
            forecast_display['date'] = forecast_display['date'].dt.date
            st.dataframe(
                forecast_display.style.format({
                    'net_forecast': '${:,.2f}',
                    'lower_ci': '${:,.2f}',
                    'upper_ci': '${:,.2f}',
                    'cumulative_forecast': '${:,.2f}'
                }),
                use_container_width=True,
                height=400
            )

# TAB 3: Enhanced Category Analysis
with tab3:
    st.markdown("<div class='section-header'>Multi-Currency Flow Analysis</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### 📥 Inflows by Currency")
        inflow_by_currency = category_analysis.get('inflow_by_currency', pd.DataFrame())
        if not inflow_by_currency.empty:
            fig_in_curr = px.pie(
                inflow_by_currency,
                values='sum',
                names='currency',
                title='Inflow Distribution (USD Equivalent)',
                color_discrete_sequence=px.colors.sequential.Blues_r,
                hole=0.4
            )
            fig_in_curr.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_in_curr, use_container_width=True)
            
            png = fig_to_png_bytes(fig_in_curr)
            if png:
                st.download_button("💾 Download", data=png, file_name="inflows_currency.png", mime="image/png", key="in_curr")
        else:
            st.info("No inflow currency data available.")
    
    with c2:
        st.markdown("#### 📤 Outflows by Currency")
        outflow_by_currency = category_analysis.get('outflow_by_currency', pd.DataFrame())
        if not outflow_by_currency.empty:
            fig_out_curr = px.pie(
                outflow_by_currency,
                values='sum',
                names='currency',
                title='Outflow Distribution (USD Equivalent)',
                color_discrete_sequence=px.colors.sequential.Reds_r,
                hole=0.4
            )
            fig_out_curr.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_out_curr, use_container_width=True)
            
            png = fig_to_png_bytes(fig_out_curr)
            if png:
                st.download_button("💾 Download", data=png, file_name="outflows_currency.png", mime="image/png", key="out_curr")
        else:
            st.info("No outflow currency data available.")
    
    st.markdown("<div class='section-header'>Category Performance Analysis</div>", unsafe_allow_html=True)
    
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("#### 📈 Top Inflow Categories")
        inflow_by_category = category_analysis.get('inflow_by_category', pd.DataFrame())
        if not inflow_by_category.empty:
            inflow_cats = inflow_by_category.sort_values(by='sum', ascending=False).head(10)
            
            fig_in_cat = px.bar(
                inflow_cats,
                x='sum',
                y='category',
                orientation='h',
                title='Top 10 Inflow Categories by Volume',
                labels={'category': 'Category', 'sum': 'Total Amount (USD)'},
                color='sum',
                color_continuous_scale='Blues'
            )
            fig_in_cat.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
            st.plotly_chart(fig_in_cat, use_container_width=True)
            
            png = fig_to_png_bytes(fig_in_cat)
            if png:
                st.download_button("💾 Download", data=png, file_name="top_inflows.png", mime="image/png", key="in_cat")
            
            # Show data table
            with st.expander("📋 View Category Details"):
                st.dataframe(
                    inflow_cats.style.format({
                        'sum': '${:,.2f}',
                        'mean': '${:,.2f}',
                        'std': '${:,.2f}',
                        'percentage': '{:.1f}%'
                    }),
                    use_container_width=True
                )
        else:
            st.info("No inflow category data available.")
    
    with colB:
        st.markdown("#### 📉 Top Outflow Categories")
        outflow_by_category = category_analysis.get('outflow_by_category', pd.DataFrame())
        if not outflow_by_category.empty:
            outflow_cats = outflow_by_category.sort_values(by='sum', ascending=False).head(10)
            
            fig_out_cat = px.bar(
                outflow_cats,
                x='sum',
                y='category',
                orientation='h',
                title='Top 10 Outflow Categories by Volume',
                labels={'category': 'Category', 'sum': 'Total Amount (USD)'},
                color='sum',
                color_continuous_scale='Reds'
            )
            fig_out_cat.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
            st.plotly_chart(fig_out_cat, use_container_width=True)
            
            png = fig_to_png_bytes(fig_out_cat)
            if png:
                st.download_button("💾 Download", data=png, file_name="top_outflows.png", mime="image/png", key="out_cat")
            
            # Show data table
            with st.expander("📋 View Category Details"):
                st.dataframe(
                    outflow_cats.style.format({
                        'sum': '${:,.2f}',
                        'mean': '${:,.2f}',
                        'std': '${:,.2f}',
                        'percentage': '{:.1f}%'
                    }),
                    use_container_width=True
                )
        else:
            st.info("No outflow category data available.")
    
    # Combined Category Comparison
    st.markdown("<div class='section-header'>Inflow vs Outflow Comparison</div>", unsafe_allow_html=True)
    
    if not inflow_by_category.empty and not outflow_by_category.empty:
        # Merge and compare
        comparison = pd.merge(
            inflow_by_category[['category', 'sum']].rename(columns={'sum': 'inflows'}),
            outflow_by_category[['category', 'sum']].rename(columns={'sum': 'outflows'}),
            on='category',
            how='outer'
        ).fillna(0)
        
        comparison['net'] = comparison['inflows'] - comparison['outflows']
        comparison = comparison.sort_values('net', ascending=False).head(10)
        
        fig_comp = go.Figure()
        
        fig_comp.add_trace(go.Bar(
            x=comparison['category'],
            y=comparison['inflows'],
            name='Inflows',
            marker_color='#3b82f6'
        ))
        
        fig_comp.add_trace(go.Bar(
            x=comparison['category'],
            y=-comparison['outflows'],
            name='Outflows',
            marker_color='#ef4444'
        ))
        
        fig_comp.update_layout(
            title='Category-wise Inflows vs Outflows',
            xaxis_title='Category',
            yaxis_title='Amount (USD)',
            barmode='relative',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)

# TAB 4: Detailed Data
with tab4:
    st.markdown("<div class='section-header'>Complete Data Export & Analysis</div>", unsafe_allow_html=True)
    
    # Summary Statistics
    with st.expander("📊 Summary Statistics", expanded=True):
        stats = generate_summary_stats(net_flows, forecast_df)
        
        col1, col2, col3 = st.columns(3)
        idx = 0
        for key, value in stats.items():
            with [col1, col2, col3][idx % 3]:
                st.markdown(f"""
                <div class='metric-item'>
                    <div class='metric-item-label'>{key}</div>
                    <div class='metric-item-value'>{value}</div>
                </div>
                """, unsafe_allow_html=True)
            idx += 1
    
    # Net Flows Data
    st.markdown("#### 💰 Net Cash Flows (Complete Dataset)")
    st.dataframe(net_flows, use_container_width=True, height=400)
    
    # Transaction Details
    if not inflows.empty or not outflows.empty:
        st.markdown("#### 📝 Transaction Details")
        
        view_col1, view_col2 = st.columns(2)
        
        with view_col1:
            view_inflows = st.checkbox("Show Inflows", value=True)
        with view_col2:
            view_outflows = st.checkbox("Show Outflows", value=True)
        
        if view_inflows and not inflows.empty:
            st.markdown("**Inflow Transactions**")
            st.dataframe(
                inflows.style.format({
                    'amount': '{:,.2f}',
                    'rate': '{:.4f}',
                    'base_amount': '${:,.2f}'
                }),
                use_container_width=True,
                height=300
            )
        
        if view_outflows and not outflows.empty:
            st.markdown("**Outflow Transactions**")
            st.dataframe(
                outflows.style.format({
                    'amount': '{:,.2f}',
                    'rate': '{:.4f}',
                    'base_amount': '${:,.2f}'
                }),
                use_container_width=True,
                height=300
            )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("""
<div class='dashboard-footer'>
    <strong>Zimbabwe Cash Flow Intelligence Platform v2.0</strong><br>
    Enhanced with AI/ML capabilities, multi-currency analysis, and advanced risk assessment<br>
    <em>💡 Tip: Install 'kaleido' package for PNG exports (pip install -U kaleido)</em>
</div>
""", unsafe_allow_html=True)
