import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from backend import ZimbabweCashFlowModel

# Page configuration
st.set_page_config(
    page_title="Zimbabwe Cash Flow Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 2rem;
    }
    .metric-container {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .risk-critical {
        color: #DC2626;
        font-weight: bold;
    }
    .risk-high {
        color: #F59E0B;
        font-weight: bold;
    }
    .risk-moderate {
        color: #10B981;
    }
    .risk-low {
        color: #0EA5E9;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='main-header'>Zimbabwe Cash Flow Simulation Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
This dashboard simulates cash flow in Zimbabwe's multi-currency environment, considering:
- ZiG (Zimbabwe Gold) - the current local currency
- USD (US Dollar) - widely used for international and some local transactions
- ZAR (South African Rand) - used for regional trade and some local transactions

The model incorporates currency volatility, payment delays, and liquidity constraints typical in Zimbabwe's economy.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Simulation Settings")
    
    # Simulation parameters
    days = st.slider("Simulation Period (Days)", min_value=30, max_value=365, value=90, step=30)
    initial_balance = st.number_input("Initial Balance (USD)", min_value=1000, max_value=1000000, value=25000, step=5000)
    
    # Currency options
    base_currency = st.selectbox("Base Currency for Reporting", ["USD", "ZiG", "ZAR"], index=0)
    
    # Advanced settings
    st.subheader("Advanced Settings")
    with st.expander("Currency Exchange Settings"):
        zig_volatility = st.slider("ZiG Volatility Factor", min_value=0.01, max_value=0.30, value=0.08, step=0.01, 
                                format="%.2f", help="Higher values increase day-to-day exchange rate fluctuations")
        zar_volatility = st.slider("ZAR Volatility Factor", min_value=0.01, max_value=0.20, value=0.03, step=0.01, 
                                format="%.2f", help="Higher values increase day-to-day exchange rate fluctuations")
    
    with st.expander("Business Settings"):
        business_scale = st.select_slider("Business Scale", 
                                        options=["Small", "Medium", "Large", "Enterprise"], 
                                        value="Medium")
        
        # Map scale to multiplier
        scale_multiplier = {
            "Small": 0.5,
            "Medium": 1.0,
            "Large": 2.5,
            "Enterprise": 5.0
        }
    
    st.subheader("Currency Information")
    st.info("""
    *ZiG (Zimbabwe Gold)* - Introduced in April 2024 as Zimbabwe's latest currency, backed by gold and other precious metals.
    
    *USD* - The US Dollar remains widely used in Zimbabwe for most major transactions.
    
    *ZAR* - South African Rand is commonly used in border regions and for import/export with South Africa.
    """)
    
    run_simulation = st.button("Run Simulation", type="primary")

# Initialize the model
model = ZimbabweCashFlowModel(
    base_currency=base_currency, 
    initial_balance=initial_balance,
    zig_volatility=zig_volatility,
    zar_volatility=zar_volatility,
    business_scale=scale_multiplier[business_scale]
)

# Run simulation when button is clicked or when parameters change
if run_simulation or 'net_flows' not in st.session_state:
    with st.spinner('Running simulation...'):
        inflows, outflows = model.simulate_transactions(days=days)
        
        # Check if we got any data
        if len(inflows) == 0 or len(outflows) == 0:
            st.error("No transaction data generated. Please try again.")
            st.stop()
        
        net_flows = model.calculate_net_flows(inflows, outflows)
        net_flows = model.analyze_liquidity(net_flows)
        
        # Process categorized data
        for idx, row in inflows.iterrows():
            currency = row['currency']
            rate = row[f'{currency}_rate']
            inflows.loc[idx, 'base_amount'] = row['amount'] / rate
            
        for idx, row in outflows.iterrows():
            currency = row['currency']
            rate = row[f'{currency}_rate']
            outflows.loc[idx, 'base_amount'] = row['amount'] / rate
            
        category_analysis = model.categorized_analysis(inflows, outflows)
        
        # Get exchange rate forecast - use the backend model directly
        # This ensures the values match what would be generated in backend.py
        exchange_forecast_model = ZimbabweCashFlowModel(
        base_currency=base_currency,
        initial_balance=initial_balance,
        zig_volatility=zig_volatility,
        zar_volatility=zar_volatility,
        business_scale=scale_multiplier[business_scale]
        )
        exchange_forecast = exchange_forecast_model.forecast_exchange_rates(days=days)
  
        # Store results in session state
        st.session_state['net_flows'] = net_flows
        st.session_state['inflows'] = inflows
        st.session_state['outflows'] = outflows
        st.session_state['category_analysis'] = category_analysis
        st.session_state['exchange_forecast'] = exchange_forecast

# Display simulation results
if 'net_flows' in st.session_state:
    net_flows = st.session_state['net_flows']
    inflows = st.session_state['inflows']
    outflows = st.session_state['outflows']
    category_analysis = st.session_state['category_analysis']
    exchange_forecast = st.session_state['exchange_forecast']
    
    st.markdown("<h2 class='section-header'>📊 Key Financial Metrics</h2>", unsafe_allow_html=True)
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "End Balance", 
            f"{net_flows['cumulative'].iloc[-1]:,.2f} {base_currency}",
            f"{net_flows['cumulative'].iloc[-1] - net_flows['cumulative'].iloc[0]:,.2f}"
        )
        
    with col2:
        avg_daily_net = net_flows['net'].mean()
        st.metric(
            "Avg. Daily Net Flow", 
            f"{avg_daily_net:,.2f} {base_currency}",
            f"{avg_daily_net / abs(net_flows['outflow'].mean()) * 100:.1f}%" if avg_daily_net > 0 else f"{avg_daily_net / abs(net_flows['outflow'].mean()) * 100:.1f}%"
        )
        
    with col3:
        max_gap = net_flows['liquidity_gap'].max()
        st.metric(
            "Max Liquidity Gap", 
            f"{max_gap:,.2f} {base_currency}",
            "0.00" if max_gap == 0 else f"{max_gap / net_flows['cumulative'].mean() * 100:.1f}% of avg balance",
            delta_color="inverse"
        )
        
    with col4:
        min_days = net_flows['days_covered'].min()
        st.metric(
            "Min. Days Covered", 
            f"{min_days:.1f} days",
            "Critical" if min_days < 7 else "Stable" if min_days > 30 else "Adequate",
            delta_color="off" if min_days > 30 else "inverse"
        )
    
    # Cash Flow Overview
    st.markdown("<h2 class='section-header'>💰 Cash Flow Overview</h2>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Cash Flow", "Cumulative Balance", "Liquidity Risk", "Currency Analysis"])
    
    with tab1:
        # Create a Plotly figure for Daily Cash Flows
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=net_flows['date'],
            y=net_flows['inflow'],
            name='Inflows',
            marker_color='rgb(26, 118, 255)'
        ))
        
        fig.add_trace(go.Bar(
            x=net_flows['date'],
            y=net_flows['outflow'] * -1,  # Make outflows negative for better visualization
            name='Outflows',
            marker_color='rgb(246, 78, 139)'
        ))
        
        fig.add_trace(go.Scatter(
            x=net_flows['date'],
            y=net_flows['net'],
            mode='lines',
            name='Net Flow',
            line=dict(color='rgb(46, 184, 46)', width=2)
        ))
        
        fig.update_layout(
            title='Daily Cash Flows',
            barmode='relative',
            bargap=0.15,
            bargroupgap=0.1,
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Create a figure for Cumulative Balance
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=net_flows['date'],
            y=net_flows['cumulative'],
            mode='lines',
            name='Total Balance',
            line=dict(color='rgb(26, 118, 255)', width=3),
            fill='tozeroy',
            fillcolor='rgba(26, 118, 255, 0.1)'
        ))
        
        fig.add_trace(go.Scatter(
            x=net_flows['date'],
            y=net_flows['available_cash'],
            mode='lines',
            name='Available Cash',
            line=dict(color='rgb(46, 184, 46)', width=2, dash='dot')
        ))
        
        fig.update_layout(
            title='Cumulative Cash Position',
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Create a figure for Liquidity Risk
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=net_flows['date'],
            y=net_flows['liquidity_gap'],
            name='Liquidity Gap',
            marker_color=np.where(net_flows['liquidity_gap'] > 0, 'rgb(220, 38, 38)', 'rgb(16, 185, 129)')
        ))
        
        fig.add_trace(go.Scatter(
            x=net_flows['date'],
            y=net_flows['days_covered'],
            mode='lines',
            name='Days Covered',
            yaxis='y2',
            line=dict(color='rgb(234, 88, 12)', width=2)
        ))
        
        fig.update_layout(
            title='Liquidity Risk Analysis',
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            yaxis=dict(title='Liquidity Gap'),
            yaxis2=dict(
                title='Days Covered',
                overlaying='y',
                side='right'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk assessment
        st.markdown("### Liquidity Risk Assessment")
        
        risk_days = net_flows['days_covered'].min()
        if risk_days < 7:
            st.error("🚨 *Critical Risk*: Cash reserves cover less than 7 days of operations")
        elif risk_days < 14:
            st.warning("⚠ *High Risk*: Cash reserves cover less than 14 days of operations")
        elif risk_days < 30:
            st.info("ℹ *Moderate Risk*: Cash reserves cover less than 30 days of operations")
        else:
            st.success("✅ *Low Risk*: Cash reserves cover more than 30 days of operations")
    
    with tab4:
        # Currency distribution
        st.markdown("### Currency Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Inflows by currency
        inflow_currency = inflows.groupby('currency')['base_amount'].sum().reset_index()
        fig = px.pie(
            inflow_currency,
            values='base_amount',
            names='currency',
            title='Inflows by Currency'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Outflows by currency
        outflow_currency = outflows.groupby('currency')['base_amount'].sum().reset_index()
        fig = px.pie(
            outflow_currency,
            values='base_amount',
            names='currency',
            title='Outflows by Currency'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Exchange rate forecast
    st.markdown("### Exchange Rate Forecast")
    if not exchange_forecast.empty:
        # Create a Plotly figure with both rates - using the consistently generated values
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=exchange_forecast['date'],
            y=exchange_forecast['ZiG_rate'],
            mode='lines',
            name='ZiG Rate (per USD)',
            line=dict(color='gold', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=exchange_forecast['date'],
            y=exchange_forecast['ZAR_rate'],
            mode='lines',
            name='ZAR Rate (per USD)',
            line=dict(color='green', width=2)
        ))
        
        # Set the y-axis range to match the ranges from the backend
        fig.update_layout(
            title='Exchange Rate Projection',
            xaxis_title='Date',
            yaxis_title='Exchange Rate (per USD)',
            height=500,
            hovermode="x unified",
            yaxis=dict(
                title='Exchange Rate',
                
                range=[17, 35]  # Slightly wider than the clipping ranges to show full data
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Information about exchange rate ranges
        st.info("""
        These ranges reflect the constrained volatility in the Zimbabwe financial markets.
        """)
    else:
        st.warning("No exchange rate forecast data available")
        
    # Category analysis
    st.markdown("<h2 class='section-header'>📈 Transaction Category Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top inflow categories
        inflow_cats = category_analysis['inflow_by_category'][('base_amount', 'sum')].sort_values(ascending=False).head(10)
        inflow_cats_df = inflow_cats.reset_index()
        inflow_cats_df.columns = ['category', 'amount']

        fig = px.bar(
            inflow_cats_df,
            x='amount',
            y='category',
            orientation='h',
            title='Top 10 Inflow Categories',
            labels={'category': 'Category', 'amount': 'Amount'}
        )

        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top outflow categories
        outflow_cats = category_analysis['outflow_by_category'][('base_amount', 'sum')].sort_values(ascending=False).head(10)
        outflow_cats_df = outflow_cats.reset_index()
        outflow_cats_df.columns = ['category', 'amount']

        fig = px.bar(
            outflow_cats_df,
            x='amount',
            y='category',
            orientation='h',
            title='Top 10 Outflow Categories',
            labels={'category': 'Category', 'amount': 'Amount'}
        )

        st.plotly_chart(fig, use_container_width=True)
    
    # Raw data
    st.markdown("<h2 class='section-header'>📋 Raw Data</h2>", unsafe_allow_html=True)
    
    with st.expander("View Net Flows Data"):
        st.dataframe(net_flows.round(2))
    
    with st.expander("View Detailed Inflows"):
        st.dataframe(inflows.round(2))
    
    with st.expander("View Detailed Outflows"):
        st.dataframe(outflows.round(2))

