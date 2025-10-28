from pathlib import Path
import base64
import io
import datetime as dt

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from backend import ZimbabweCashFlowModel

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title=" MWC CashflowFlow: AI-Powered Treasury Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
#   Dark UI CSS
# ----------------------------
 _CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap');
/* Load Font Awesome for Icons */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css');

:root {
    --bg-dark: #0f172a; /* Slate 900 */
    --card-bg: rgba(30, 41, 59, 0.7); /* Slate 800 with transparency for glass effect */
    --card-border: rgba(255, 255, 255, 0.1);
    --text-light: #f8fafc;
    --text-muted: #94a3b8;
    --primary-color: #3b82f6; /* Blue 500 */
    --success-color: #10b981; /* Emerald 500 */
    --danger-color: #ef4444; /* Red 500 */
    --warning-color: #f59e0b; /* Amber 500 */
    -- -gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --quantum-gradient: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
}

/* Base Body Styling - IMPORTANT for Streamlit to adopt the dark background */
.stApp {
    background-color: var(--bg-dark);
    color: var(--text-light);
    font-family: 'Inter', sans-serif;
}

/* Streamlit Main Content */
.main {
    background-color: var(--bg-dark);
    color: var(--text-light);
}

/* ---   HEADER IMPLEMENTATION --- */
. -header-container {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 1.2rem 2rem;
    margin-bottom: 0;
    
    /*   Sticky Properties */
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 99999 !important;
    
    /*   Gradient Background */
    background: var(--quantum-gradient);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    width: 100%;
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.15);
}

/* Logo Container with   Styling */
.logo-container {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 70px;
    height: 70px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

/* Header Content */
.header-content {
    display: flex;
    flex-direction: column;
    flex: 1;
}

.system-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.8rem !important;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    margin-bottom: 4px !important;
    line-height: 1.1;
}

.system-tagline {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem !important;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.9);
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    letter-spacing: 0.5px;
    margin: 0;
}

/* Status Badge */
.status-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 255, 255, 0.15);
    padding: 8px 16px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    font-size: 0.85rem;
    font-weight: 600;
    color: white;
}

.live-indicator {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

/* Push down main content to clear the fixed header */
.block-container {
    padding-top: 7rem !important;
    padding-bottom: 2rem;
}

/* Adjust Streamlit's wrapper element if necessary */
div[data-testid="stVerticalBlock"]:first-child {
    margin-top: 0 !important;
}
/* ----------------------------------------------------- */

/* Custom Metric Cards */
.metric-card {
    background-color: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(6px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.6);
    transition: transform 0.2s;
    height: 100%; /* Ensure all cards are the same height */
}

.metric-card:hover {
    transform: translateY(-3px);
}

.metric-card .icon-label {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
    color: var(--text-muted);
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
}

.metric-card .icon {
    font-size: 1.35rem;
    color: var(--primary-color); /* Default icon color */
}

.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1.1;
    color: var(--text-light);
    margin-bottom: 0.35rem;
}

.metric-card .sub-text {
    font-size: 0.875rem;
    color: var(--text-muted);
}

/* Status Colors for Values */
.status-positive { color: var(--success-color) !important; }
.status-negative { color: var(--danger-color) !important; }
.status-neutral { color: var(--text-light) !important; }
.status-warning-text { color: var(--warning-color) !important; }

/* Chart and Data Table Section */
.chart-section, .data-section {
    background-color: rgba(30,41,59,0.72);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 18px;
    margin-top: 1.2rem;
    backdrop-filter: blur(6px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.6);
}

/* Streamlit Tabs/Expander Headers */
.stTabs [data-baseweb="tab-list"] button {
    background-color: rgba(30,41,59,0.72);
    color: var(--text-light);
    border-radius: 8px 8px 0 0 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-bottom: none !important;
    font-weight: 600;
}

/* Streamlit sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(10, 20, 40, 0.95);
}

/* Fix for Streamlit text and widgets in dark mode */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-light);
}
p, .stMarkdown, label {
    color: var(--text-light);
}

/* Footer */
.dashboard-footer {
    text-align: center;
    padding: 1rem;
    color: var(--text-muted);
    font-size: 12px;
    margin-top: 1.5rem;
}

/* ---   BUTTON STYLING (General) --- */
div.stButton > button {
    background-color: var(--primary-color) !important;
    color: var(--text-light) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
    transition: all 0.2s !important;
}

div.stButton > button:hover {
    background-color: #2563eb !important; /* Blue 600 */
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(59, 130, 246, 0.35) !important;
}

div.stButton > button:active {
    background-color: #1e40af !important; /* Blue 800 */
    transform: translateY(0);
}

/* Specific styling for the 'Download' buttons in tab4 */
div.stDownloadButton > button {
    background-color: var(--success-color) !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25) !important;
}

div.stDownloadButton > button:hover {
    background-color: #059669 !important; /* Emerald 600 */
    box-shadow: 0 6px 16px rgba(16, 185, 129, 0.35) !important;
}

/* ---   BUTTON STYLING (Sidebar Form) --- */
[data-testid="stForm"] div.stButton button {
    width: 100%;
    margin-top: 10px;
    /* CHANGE TEXT COLOR TO PURE BLACK FOR DEFINITE HIGH CONTRAST */
    color: #000000 !important; 
}
/* ------------------------------------------------ */

</style>
"""
st.markdown( _CSS, unsafe_allow_html=True)

# ----------------------------
# Helper functions (merged)
# ----------------------------
def display_logo(filename="logo.svg", width=56, alt="Logo"):
    """Robust logo helper (embed svg or fallback to image/text)."""
    try:
        base_path = Path(__file__).parent
    except NameError:
        base_path = Path.cwd()
    logo_path = (base_path / filename)
    if not logo_path.exists():
        for f in base_path.iterdir():
            if f.name.lower() == filename.lower():
                logo_path = f
                break
    if not logo_path.exists():
        st.markdown(f"<div style='font-weight:700; font-size:18px'>{alt}</div>", unsafe_allow_html=True)
        return

    if logo_path.suffix.lower() == ".svg":
        try:
            svg_text = logo_path.read_text(encoding="utf-8")
            b64 = base64.b64encode(svg_text.encode("utf-8")).decode("utf-8")
            img_tag = f'<img src="data:image/svg+xml;base64,{b64}" width="{width}" alt="{alt}"/>'
            st.markdown(img_tag, unsafe_allow_html=True)
            return
        except Exception:
            pass

    try:
        st.image(str(logo_path), width=width)
    except Exception:
        st.markdown(f"<div style='font-weight:700; font-size:18px'>{alt}</div>", unsafe_allow_html=True)


def style_plotly_figure(fig, title_text=None, height=450):
    """Apply dark theme to Plotly figure consistent with UI."""
    if title_text:
        fig.update_layout(
            title=dict(text=title_text, x=0.5, xanchor='center', font=dict(color='white', size=16)),
        )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', title_font=dict(color='#94a3b8'), tickfont=dict(color='#94a3b8')),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', title_font=dict(color='#94a3b8'), tickfont=dict(color='#94a3b8')),
        legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0)')
    )
    fig.update_traces(hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter"))
    return fig

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
#   HEADER with   Branding
# ----------------------------
st.markdown("<div class=' -header-container'>", unsafe_allow_html=True)

# Logo in   container
st.markdown("<div class='logo-container'>", unsafe_allow_html=True)
display_logo("logo.svg", width=48, alt="MWC")
st.markdown("</div>", unsafe_allow_html=True)

# Header content with new system name and tagline
st.markdown("""
<div class='header-content'>
    <div class='system-name'> MWC CashflowFlow</div>
    <div class='system-tagline'>AI-Powered Treasury Intelligence Platform • Multi-Currency Risk Analytics</div>
</div>
""", unsafe_allow_html=True)

# Status badge
st.markdown("""
<div class='status-badge'>
    <div class='live-indicator'></div>
    <span>LIVE ANALYSIS</span>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Sidebar: controls from front2 but styled here
# ----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Simulation Configuration")
    with st.form("controls"):
        st.markdown("#### Financial Parameters")
        initial_balance = st.number_input("Initial USD Balance", value=25000, min_value=0, step=500, format="%d")
        st.markdown("#### Time Parameters")
        col1, col2 = st.columns(2)
        with col1:
            sim_days = st.number_input("Historical Days", min_value=30, max_value=365, value=180, step=30)
        with col2:
            forecast_days = st.number_input("Forecast Days", min_value=15, max_value=90, value=30, step=15)
        st.markdown("#### Business & Volatility")
        business_scale = st.slider("Business Scale Multiplier", 0.5, 3.0, 1.0, 0.1)
        col3, col4 = st.columns(2)
        with col3:
            zig_volatility = st.slider("ZiG Volatility", 0.005, 0.15, 0.02, 0.001)
        with col4:
            zar_volatility = st.slider("ZAR Volatility", 0.002, 0.07, 0.01, 0.001)

        # Advanced controls (not nested inside another expander)
        st.markdown("---")
        st.markdown("#### Advanced")
        seed_option = st.checkbox("Fixed Random Seed", value=False)
        seed_value = st.number_input("Seed Value", value=42, step=1) if seed_option else None
        car_confidence = st.slider("CaR Confidence Level", min_value=0.90, max_value=0.999, value=0.95, step=0.005)
        arima_order_text = st.text_input("ARIMA order (p,d,q)", value="5,1,0")

        presentation_mode = st.checkbox("Presentation Mode", value=False)

        col_run, col_reset = st.columns(2)
        with col_run:
            # The text color for this button is now darker via CSS
            run = st.form_submit_button("▶️ Run Analysis")
        with col_reset:
            # The text color for this button is now darker via CSS
            reset = st.form_submit_button("🔄 Reset View")

    st.markdown("---")
    st.markdown("### 📚 Quick Guide")
    st.markdown("""
    1. Configure parameters above
    2. Click Run Analysis to generate simulation
    3. Explore visualizations & export data
    """)

# Reset handling
if 'reset' in locals() and reset:
    for k in ['run_model','net_flows','forecast_df','category_analysis','inflows','outflows','params']:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

# Persist params to session
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
    st.markdown("<style>.system-name{font-size:3rem !important}</style>", unsafe_allow_html=True)

if run:
    st.session_state['run_model'] = True

# ----------------------------
# Run model when requested (front2 functionality)
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

    model = ZimbabweCashFlowModel(
        initial_balance=float(params['initial_balance']),
        zig_volatility=float(params['zig_volatility']),
        zar_volatility=float(params['zar_volatility']),
        business_scale=float(params['business_scale'])
    )

    with st.spinner("Running simulation & AI analysis..."):
        pbar = st.progress(0)
        pbar.progress(10)
        inflows, outflows = model.simulate_transactions(days=int(params['sim_days']))
        pbar.progress(40)
        net_flows = model.calculate_net_flows(inflows, outflows)
        pbar.progress(60)

        if net_flows.empty:
            st.error("Simulation produced no cash flows. Adjust parameters and retry.")
            st.session_state['run_model'] = False
            st.stop()

        net_flows = model.analyze_liquidity(net_flows)
        net_flows_with_car = model.cash_at_risk_analysis(net_flows, confidence_level=float(params.get('car_confidence', 0.95)))
        pbar.progress(80)

        forecast_df = model.ml_forecast_net_flow(net_flows_with_car, forecast_days=int(params['forecast_days']), order=arima_order)
        category_analysis = model.categorized_analysis(inflows, outflows)
        pbar.progress(100)

        # normalize date columns
        net_flows_with_car['date'] = pd.to_datetime(net_flows_with_car['date'])
        if not forecast_df.empty:
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])

        st.session_state['net_flows'] = net_flows_with_car
        st.session_state['forecast_df'] = forecast_df
        st.session_state['category_analysis'] = category_analysis
        st.session_state['inflows'] = inflows
        st.session_state['outflows'] = outflows
        st.session_state['run_model'] = True # Ensure model ran successfully
        st.success("Analysis complete!")

# ----------------------------
# If no results, show CTA
# ----------------------------
if 'net_flows' not in st.session_state or st.session_state['net_flows'].empty or not st.session_state.get('run_model', False):
    st.markdown("""
    <div class='chart-section'>
      <h3 style='margin:0 0 8px 0'>Welcome to  MWC CashflowFlow Treasury Intelligence</h3>
      <p style='color:#94a3b8'>Configure simulation parameters in the sidebar and click <strong>Run Analysis</strong> to generate cash flow simulations, Cash-at-Risk (CaR) analysis, and AI/ARIMA forecasts.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load results from session
net_flows = st.session_state['net_flows']
forecast_df = st.session_state.get('forecast_df', pd.DataFrame())
category_analysis = st.session_state.get('category_analysis', {})
inflows = st.session_state.get('inflows', pd.DataFrame())
outflows = st.session_state.get('outflows', pd.DataFrame())
params = st.session_state['params']


# ----------------------------
# This content is correctly pushed down by the .block-container padding-top change
# ----------------------------
metric_cols = st.columns(4)

with metric_cols[0]:
    total_inflows = net_flows['inflow'].sum() if 'inflow' in net_flows.columns else 0.0
    render_html = f"""
    <div class='metric-card'>
        <div class='icon-label'><div class='icon' style='color:#10b981'><i class='fa-solid fa-wallet'></i></div><span>Total Inflows</span></div>
        <div class='value status-positive'>${total_inflows:,.2f}</div>
        <div class='sub-text'>Total inflows (USD equivalent)</div>
    </div>
    """
    st.markdown(render_html, unsafe_allow_html=True)

with metric_cols[1]:
    current_balance = float(net_flows['cumulative'].iloc[-1])
    # Apply success color if balance is > initial balance, else neutral
    balance_status_class = "status-positive" if current_balance >= params['initial_balance'] else "status-neutral"
    render_html = f"""
    <div class='metric-card'>
        <div class='icon-label'><div class='icon' style='color:#3b82f6'><i class='fa-solid fa-coins'></i></div><span>Current Balance</span></div>
        <div class='value {balance_status_class}'>${current_balance:,.2f}</div>
        <div class='sub-text'>Latest cumulative balance (USD)</div>
    </div>
    """
    st.markdown(render_html, unsafe_allow_html=True)

with metric_cols[2]:
    avg_daily_net = float(net_flows['net'].mean())
    # Apply status class based on sign
    net_status_class = "status-positive" if avg_daily_net > 0 else "status-negative" if avg_daily_net < 0 else "status-neutral"
    render_html = f"""
    <div class='metric-card'>
        <div class='icon-label'><div class='icon' style='color:#f59e0b'><i class='fa-solid fa-chart-line'></i></div><span>Avg Daily Net</span></div>
        <div class='value {net_status_class}'>${avg_daily_net:,.2f}</div>
        <div class='sub-text'>Average daily net flow (USD)</div>
    </div>
    """
    st.markdown(render_html, unsafe_allow_html=True)

with metric_cols[3]:
    worst_case = float(net_flows['worst_case_cumulative'].iloc[-1]) if 'worst_case_cumulative' in net_flows.columns else current_balance
    render_html = f"""
    <div class='metric-card'>
        <div class='icon-label'><div class='icon' style='color:#ef4444'><i class='fa-solid fa-shield-halved'></i></div><span>Cash-at-Risk</span></div>
        <div class='value status-negative'>${worst_case:,.2f}</div>
        <div class='sub-text'>Worst-case cumulative balance (USD)</div>
    </div>
    """
    st.markdown(render_html, unsafe_allow_html=True)

# ----------------------------
# Main tabs & charts (preserve front2 displays)
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Cash Flow Analysis", "🧠 AI Forecasting", "📊 Category Breakdown", "📋 Data & Exports"])

# TAB 1: Cash Flow Analysis
with tab1:
    st.markdown("<div class='chart-section'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0 0 12px 0'>Cumulative Balance & Risk Timeline</h3>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=net_flows['date'],
        y=net_flows['cumulative'],
        mode='lines',
        name='Actual Balance',
        line=dict(color='#3b82f6', width=3)
    ))
    if 'worst_case_cumulative' in net_flows.columns:
        fig.add_trace(go.Scatter(
            x=net_flows['date'],
            y=net_flows['worst_case_cumulative'],
            mode='lines',
            name='Worst-Case (CaR)',
            line=dict(color='#ef4444', width=2, dash='dash')
        ))

    fig.update_layout(xaxis_title='Date', yaxis_title='Amount (USD)', hovermode='x unified', height=380)
    fig = style_plotly_figure(fig, title_text='Simulated Cumulative Cash Flow vs Worst-Case')

    st.plotly_chart(fig, use_container_width=True)

    # Daily net flow
    st.markdown("<h4 style='margin-top:16px'>Daily Net Flow</h4>", unsafe_allow_html=True)
    fig_net = go.Figure()
    colors = ['#10b981' if x >= 0 else '#ef4444' for x in net_flows['net']]
    fig_net.add_trace(go.Bar(x=net_flows['date'], y=net_flows['net'], marker_color=colors, name='Daily Net Flow'))
    fig_net.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    fig_net = style_plotly_figure(fig_net, title_text='Daily Net Cash Flow', height=320)
    st.plotly_chart(fig_net, use_container_width=True)

    # Recent CaR table
    st.markdown("<h4 style='margin-top:16px'>Recent Cash-at-Risk Analysis</h4>", unsafe_allow_html=True)
    car_cols = ['date', 'cumulative']
    if 'cash_at_risk' in net_flows.columns: car_cols.append('cash_at_risk')
    if 'worst_case_cumulative' in net_flows.columns: car_cols.append('worst_case_cumulative')
    if 'worst_case_risk' in net_flows.columns: car_cols.append('worst_case_risk')
    if 'liquidity_risk' in net_flows.columns: car_cols.append('liquidity_risk')

    display_df = net_flows[car_cols].tail(25).copy()
    display_df['date'] = display_df['date'].dt.date
    st.dataframe(display_df.style.format({
        'cumulative': '${:,.2f}',
        'cash_at_risk': '${:,.2f}',
        'worst_case_cumulative': '${:,.2f}'
    }), use_container_width=True, height=300)
    st.markdown("</div>", unsafe_allow_html=True)

# TAB 2: Forecasting
with tab2:
    st.markdown("<div class='chart-section'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0 0 12px 0'>AI/ML Forecasting (ARIMA)</h3>", unsafe_allow_html=True)

    if forecast_df is None or forecast_df.empty:
        st.info("Forecast unavailable. ARIMA may have failed or there's insufficient data.")
    else:
        # Forecast with CI
        fig_fore = go.Figure()
        fig_fore.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['upper_ci'], mode='lines', line=dict(width=0), showlegend=False))
        fig_fore.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['lower_ci'], mode='lines', fill='tonexty', fillcolor='rgba(59,130,246,0.15)', name='95% CI', line=dict(width=0)))
        fig_fore.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['net_forecast'], mode='lines', name='Net Flow Forecast', line=dict(color='#3b82f6', width=3)))
        fig_fore = style_plotly_figure(fig_fore, title_text='Daily Net Flow Forecast', height=420)
        st.plotly_chart(fig_fore, use_container_width=True)

        # cumulative forecast
        historical_data = net_flows[['date','cumulative']].rename(columns={'cumulative':'cumulative_forecast'})
        combined_df = pd.concat([historical_data, forecast_df[['date','cumulative_forecast']]]).reset_index(drop=True)
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=historical_data['date'], y=historical_data['cumulative_forecast'], mode='lines', name='Historical', line=dict(color='#0b5fa8', width=3)))
        fig_cum.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['cumulative_forecast'], mode='lines', name='Forecast', line=dict(color='#10b981', width=3, dash='dash')))
        last_hist_date = to_pydatetime_safe(historical_data['date'].iloc[-1])
        y_min = float(combined_df['cumulative_forecast'].min())
        y_max = float(combined_df['cumulative_forecast'].max())
        fig_cum.add_shape(type="line", x0=last_hist_date, x1=last_hist_date, y0=y_min, y1=y_max, line=dict(color="#f59e0b", dash="dash", width=2))
        fig_cum.add_annotation(x=last_hist_date, y=y_max, text="Forecast Start", showarrow=True, arrowhead=1, ax=0, ay=-30, bgcolor="rgba(245,158,11,0.1)")
        fig_cum = style_plotly_figure(fig_cum, title_text='Combined Historical & Forecast Cumulative Balance', height=420)
        st.plotly_chart(fig_cum, use_container_width=True)

        # Forecast table expander
        with st.expander("View Forecast Table", expanded=False):
            ft = forecast_df.copy()
            ft['date'] = ft['date'].dt.date
            st.dataframe(ft.style.format({
                'net_forecast': '${:,.2f}',
                'lower_ci': '${:,.2f}',
                'upper_ci': '${:,.2f}',
                'cumulative_forecast': '${:,.2f}'
            }), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# TAB 3: Category Breakdown
with tab3:
    st.markdown("<div class='chart-section'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0 0 12px 0'>Multi-Currency & Category Analysis</h3>", unsafe_allow_html=True)

    inflow_by_currency = category_analysis.get('inflow_by_currency', pd.DataFrame())
    outflow_by_currency = category_analysis.get('outflow_by_currency', pd.DataFrame())
    inflow_by_category = category_analysis.get('inflow_by_category', pd.DataFrame())
    outflow_by_category = category_analysis.get('outflow_by_category', pd.DataFrame())

    c1, c2 = st.columns(2)
    with c1:
        if not inflow_by_currency.empty:
            fig_in = px.pie(inflow_by_currency, values='sum', names='currency', title='Inflows by Currency (USD eq.)', hole=0.4)
            fig_in = style_plotly_figure(fig_in, title_text=None, height=360)
            st.plotly_chart(fig_in, use_container_width=True)
        else:
            st.info("No inflow currency data.")
    with c2:
        if not outflow_by_currency.empty:
            fig_out = px.pie(outflow_by_currency, values='sum', names='currency', title='Outflows by Currency (USD eq.)', hole=0.4)
            fig_out = style_plotly_figure(fig_out, title_text=None, height=360)
            st.plotly_chart(fig_out, use_container_width=True)
        else:
            st.info("No outflow currency data.")

    st.markdown("<h4 style='margin-top:14px'>Top Categories</h4>", unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        if not inflow_by_category.empty:
            inflow_cats = inflow_by_category.sort_values('sum', ascending=False).head(10)
            fig3 = px.bar(inflow_cats, x='sum', y='category', orientation='h', title='Top Inflow Categories', labels={'sum':'Amount (USD)','category':'Category'}, color='sum', color_continuous_scale='Blues')
            fig3 = style_plotly_figure(fig3, title_text=None, height=360)
            st.plotly_chart(fig3, use_container_width=True)
    with colB:
        if not outflow_by_category.empty:
            outflow_cats = outflow_by_category.sort_values('sum', ascending=False).head(10)
            fig4 = px.bar(outflow_cats, x='sum', y='category', orientation='h', title='Top Outflow Categories', labels={'sum':'Amount (USD)','category':'Category'}, color='sum', color_continuous_scale='Reds')
            fig4 = style_plotly_figure(fig4, title_text=None, height=360)
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# TAB 4: Data & Exports
with tab4:
    st.markdown("<div class='data-section'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0 0 12px 0'>Data & Export</h3>", unsafe_allow_html=True)

    stats = generate_summary_stats(net_flows, forecast_df)
    cols_stats = st.columns(3)
    i = 0
    for k, v in stats.items():
        with cols_stats[i % 3]:
            st.markdown(f"<div style='padding:10px; border-radius:8px; background:rgba(255,255,255,0.03)'><strong>{k}</strong><div style='color:#94a3b8'>{v}</div></div>", unsafe_allow_html=True)
        i += 1

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        # Download button uses custom CSS now
        st.download_button("Download Net Flows CSV", data=df_to_csv_bytes(net_flows), file_name="net_flows.csv", mime="text/csv")
    with dl2:
        if not forecast_df.empty:
            st.download_button("Download Forecast CSV", data=df_to_csv_bytes(forecast_df), file_name="forecast.csv", mime="text/csv")
        else:
            st.button("No Forecast to Download", disabled=True)
    with dl3:
        combined = pd.concat([inflows.assign(type='inflow'), outflows.assign(type='outflow')]) if (not inflows.empty or not outflows.empty) else pd.DataFrame()
        if not combined.empty:
            st.download_button("Download Transactions CSV", data=df_to_csv_bytes(combined), file_name="transactions.csv", mime="text/csv")
        else:
            st.button("No Transactions Data", disabled=True)

    st.markdown("<h4 style='margin-top:12px'>Net Flows (full table)</h4>", unsafe_allow_html=True)
    st.dataframe(net_flows, use_container_width=True, height=360)

    # Transaction details toggles
    if (not inflows.empty) or (not outflows.empty):
        colv1, colv2 = st.columns(2)
        with colv1:
            show_inflows = st.checkbox("Show Raw Inflows", value=True)
        with colv2:
            show_outflows = st.checkbox("Show Raw Outflows", value=True)

        if show_inflows and not inflows.empty:
            st.markdown("<h5 style='color:#10b981'>Inflows</h5>", unsafe_allow_html=True)
            st.dataframe(inflows.style.format({'amount':'{:,.2f}','rate':'{:.4f}','base_amount':'${:,.2f}'}), use_container_width=True, height=240)

        if show_outflows and not outflows.empty:
            st.markdown("<h5 style='color:#ef4444'>Outflows</h5>", unsafe_allow_html=True)
            st.dataframe(outflows.style.format({'amount':'{:,.2f}','rate':'{:.4f}','base_amount':'${:,.2f}'}), use_container_width=True, height=240)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div class='dashboard-footer'> MWC CashflowFlow Treasury Intelligence • AI-Powered Risk Analytics Platform</div>", unsafe_allow_html=True)



