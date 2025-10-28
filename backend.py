import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm

class ZimbabweCashFlowModel:
    def __init__(
        self,
        base_currency: str = 'USD',
        initial_balance: float = 25000,
        zig_volatility: float = 0.02,
        zar_volatility: float = 0.01,
        business_scale: float = 1.0
    ):
        self.base_currency = base_currency
        self.initial_balance = initial_balance
        self.zig_volatility = zig_volatility
        self.zar_volatility = zar_volatility
        self.business_scale = business_scale

        # Updated base rates
        self.base_zig_rate = 26.8337
        self.base_zar_rate = 18.25

        self.exchange_rates = {
            'USD': 1.0,
            'ZiG': self.base_zig_rate,
            'ZAR': self.base_zar_rate
        }

        self.payment_delays = {
            'local_ZiG': {'probability': 0.25, 'mean_days': 5, 'std_days': 3},
            'local_USD': {'probability': 0.10, 'mean_days': 2, 'std_days': 1},
            'international_USD': {'probability': 0.40, 'mean_days': 7, 'std_days': 3},
            'regional_ZAR': {'probability': 0.20, 'mean_days': 3, 'std_days': 2}
        }

        self.cash_availability = {
            'USD': 0.85,
            'ZiG': 0.75,
            'ZAR': 0.80
        }

        self.transaction_categories = {
            'inflows': {
                'sales_revenue': {'weight': 0.5, 'volatility': 0.15, 'currencies': ['ZiG', 'USD', 'ZAR'], 'distribution': [0.60, 0.30, 0.10]},
                'service_fees': {'weight': 0.2, 'volatility': 0.10, 'currencies': ['ZiG', 'USD'], 'distribution': [0.70, 0.30]},
                'investments': {'weight': 0.15, 'volatility': 0.40, 'currencies': ['USD'], 'distribution': [1.0]},
                'loans': {'weight': 0.1, 'volatility': 0.05, 'currencies': ['USD', 'ZAR'], 'distribution': [0.8, 0.2]},
                'grants': {'weight': 0.05, 'volatility': 0.50, 'currencies': ['USD'], 'distribution': [1.0]}
            },
            'outflows': {
                'supplier_payments': {'weight': 0.35, 'volatility': 0.12, 'currencies': ['ZiG', 'USD', 'ZAR'], 'distribution': [0.50, 0.30, 0.20]},
                'salaries': {'weight': 0.25, 'volatility': 0.05, 'currencies': ['ZiG', 'USD'], 'distribution': [0.80, 0.20]},
                'rent_utilities': {'weight': 0.15, 'volatility': 0.08, 'currencies': ['ZiG'], 'distribution': [1.0]},
                'tax_payments': {'weight': 0.15, 'volatility': 0.10, 'currencies': ['ZiG'], 'distribution': [1.0]},
                'capital_expenses': {'weight': 0.10, 'volatility': 0.30, 'currencies': ['USD', 'ZAR'], 'distribution': [0.70, 0.30]}
            }
        }

        self.seasonality = {
            'monthly': {1: 0.9, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.05, 6: 1.1,
                        7: 1.15, 8: 1.2, 9: 1.15, 10: 1.1, 11: 1.2, 12: 1.3},
            'weekly': {0: 0.9, 1: 1.0, 2: 1.1, 3: 1.15, 4: 1.2, 5: 0.8, 6: 0.7}
        }

    # --- Helper Methods ---
    def _get_current_zig_rate(self):
        volatility = np.random.normal(0, self.zig_volatility)
        return self.base_zig_rate * (1 + volatility)

    def _get_current_zar_rate(self):
        volatility = np.random.normal(0, self.zar_volatility)
        return self.base_zar_rate * (1 + volatility)

    def _update_exchange_rates(self, date: datetime):
        zig_trend = np.sin(date.day / 10) * 0.02
        zig_random = np.random.normal(0, self.zig_volatility / 2)
        zar_trend = np.sin(date.day / 15) * 0.01
        zar_random = np.random.normal(0, self.zar_volatility / 2)

        self.exchange_rates['ZiG'] *= (1 + zig_trend + zig_random)
        self.exchange_rates['ZAR'] *= (1 + zar_trend + zar_random)

        # Keep USD fixed as base
        self.exchange_rates['USD'] = 1.0
        return self.exchange_rates.copy()

    # --- Simulation ---
    def simulate_transactions(self, days: int = 30, categorized: bool = True):
        """Simulate cash flows with realistic Zimbabwe market conditions.
        Returns two DataFrames: inflows and outflows. Each has:
        - date (original event date),
        - expected_date (when cash is actually available/paid),
        - category, currency, amount (in currency units), currency_rate columns, base_amount (USD)
        """
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        dates = [start_date + timedelta(days=i) for i in range(days)]

        base_inflow = 12000 * self.business_scale
        base_outflow = 10000 * self.business_scale

        all_inflows = []
        all_outflows = []

        for day in dates:
            month_factor = self.seasonality['monthly'][day.month]
            weekday_factor = self.seasonality['weekly'][day.weekday()]
            seasonal_factor = month_factor * weekday_factor

            daily_rates = self._update_exchange_rates(day)

            # Inflows
            for category, params in self.transaction_categories['inflows'].items():
                category_base = base_inflow * params['weight'] * seasonal_factor
                category_amount = category_base * (1 + np.random.normal(0, params['volatility']))

                for i, currency in enumerate(params['currencies']):
                    if np.random.random() < 0.7:
                        amount = category_amount * params['distribution'][i] * np.random.uniform(0.9, 1.1)
                        # Convert amount into currency units (if currency != base, convert using rate)
                        currency_amount = amount * daily_rates[currency] if currency != self.base_currency else amount

                        # Determine expected/settlement date using delay model
                        if currency == 'ZiG':
                            delay_key = 'local_ZiG'
                        elif currency == 'ZAR':
                            delay_key = 'regional_ZAR'
                        else:
                            # USD inflows may be local or international by probability (we use local_USD prob defined)
                            delay_key = 'international_USD' if np.random.random() < 0.6 else 'local_USD'
                        delay_params = self.payment_delays.get(delay_key, self.payment_delays['local_ZiG'])
                        delay_days = 0
                        if np.random.random() < delay_params['probability']:
                            delay_days = int(max(0, np.random.normal(delay_params['mean_days'], delay_params['std_days'])))

                        row = {
                            'date': day,
                            'category': category,
                            'currency': currency,
                            'amount': max(0.0, currency_amount),
                            f'{currency}_rate': daily_rates[currency],
                            'expected_date': day + timedelta(days=delay_days)
                        }
                        all_inflows.append(row)

            # Outflows
            for category, params in self.transaction_categories['outflows'].items():
                category_base = base_outflow * params['weight'] * seasonal_factor
                category_amount = category_base * (1 + np.random.normal(0, params['volatility']))

                for i, currency in enumerate(params['currencies']):
                    if np.random.random() < 0.8:
                        amount = category_amount * params['distribution'][i] * np.random.uniform(0.95, 1.05)
                        currency_amount = amount * daily_rates[currency] if currency != self.base_currency else amount

                        # special adjustments
                        if category == 'salaries' and day.day == 25:
                            currency_amount *= 1.2
                        if category == 'tax_payments' and day.day == 15 and day.month in [3, 6, 9, 12]:
                            currency_amount *= 3.0

                        row = {
                            'date': day,
                            'category': category,
                            'currency': currency,
                            'amount': max(0.0, currency_amount),
                            f'{currency}_rate': daily_rates[currency],
                            # outflows are assumed paid on the simulation date
                            'expected_date': day
                        }
                        all_outflows.append(row)

        inflows = pd.DataFrame(all_inflows)
        outflows = pd.DataFrame(all_outflows)

        # If no rows, return empty DataFrames with expected schema
        if inflows.empty:
            inflows = pd.DataFrame(columns=['date', 'category', 'currency', 'amount', 'expected_date'])
        if outflows.empty:
            outflows = pd.DataFrame(columns=['date', 'category', 'currency', 'amount', 'expected_date'])

        # Vectorized extraction of rate: each row has a currency-specific rate column (e.g., 'ZiG_rate').
        def extract_rate(df):
            if df.empty:
                return df
            # Ensure date columns are proper datetimes
            df['date'] = pd.to_datetime(df['date'])
            df['expected_date'] = pd.to_datetime(df['expected_date'])
            # find rate value per row
            def row_rate(r):
                key = f"{r['currency']}_rate"
                return r.get(key, 1.0) if key in r.index else 1.0
            df['rate'] = df.apply(row_rate, axis=1)
            # base_amount is amount converted back to USD (or base_currency)
            df['base_amount'] = df['amount'] / df['rate'].replace({0: np.nan})
            df['base_amount'] = df['base_amount'].fillna(0.0)
            return df

        inflows = extract_rate(inflows)
        outflows = extract_rate(outflows)

        # Keep consistent column order
        inflows = inflows[['date', 'expected_date', 'category', 'currency', 'amount', 'rate', 'base_amount']].copy()
        outflows = outflows[['date', 'expected_date', 'category', 'currency', 'amount', 'rate', 'base_amount']].copy()

        return inflows, outflows

    # --- Net Flows ---
    def calculate_net_flows(self, inflows: pd.DataFrame, outflows: pd.DataFrame):
        """Calculate daily net flows (in base currency) using expected_date as the settlement date."""
        if (inflows is None or inflows.empty) and (outflows is None or outflows.empty):
            return pd.DataFrame()

        # Determine date range from expected settlement dates (cash availability)
        dates = []
        if not inflows.empty:
            inflows['expected_date'] = pd.to_datetime(inflows['expected_date']).dt.normalize()
            dates.append(inflows['expected_date'].min())
            dates.append(inflows['expected_date'].max())
        if not outflows.empty:
            outflows['expected_date'] = pd.to_datetime(outflows['expected_date']).dt.normalize()
            dates.append(outflows['expected_date'].min())
            dates.append(outflows['expected_date'].max())

        start_date = min(d for d in dates if pd.notnull(d))
        end_date = max(d for d in dates if pd.notnull(d))
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        net_flows = pd.DataFrame({'date': date_range})
        net_flows['date'] = pd.to_datetime(net_flows['date']).dt.normalize()

        # Aggregate by expected_date (when cash arrives / leaves)
        if not inflows.empty:
            in_grouped = inflows.groupby(inflows['expected_date'].dt.normalize())['base_amount'].sum().reset_index()
            in_grouped.rename(columns={'expected_date': 'date', 'base_amount': 'inflow'}, inplace=True)
        else:
            in_grouped = pd.DataFrame(columns=['date', 'inflow'])

        if not outflows.empty:
            out_grouped = outflows.groupby(outflows['expected_date'].dt.normalize())['base_amount'].sum().reset_index()
            out_grouped.rename(columns={'expected_date': 'date', 'base_amount': 'outflow'}, inplace=True)
        else:
            out_grouped = pd.DataFrame(columns=['date', 'outflow'])

        # Merge with full date index
        net_flows = net_flows.merge(in_grouped, on='date', how='left').merge(out_grouped, on='date', how='left')
        net_flows['inflow'] = net_flows['inflow'].fillna(0.0)
        net_flows['outflow'] = net_flows['outflow'].fillna(0.0)

        net_flows['net'] = net_flows['inflow'] - net_flows['outflow']
        net_flows['cumulative'] = float(self.initial_balance) + net_flows['net'].cumsum()

        # keep date sorted and normalized
        net_flows = net_flows.sort_values('date').reset_index(drop=True)
        net_flows['date'] = pd.to_datetime(net_flows['date'])

        return net_flows

    # --- Liquidity Analysis ---
    def analyze_liquidity(self, net_flows: pd.DataFrame):
        """Add liquidity metrics: available_cash, days_covered and liquidity_risk."""
        if net_flows is None or net_flows.empty:
            return pd.DataFrame()

        df = net_flows.copy()
        availability_factor = 0.5884
        df['available_cash'] = df['cumulative'] * availability_factor

        daily_avg_outflow = df['outflow'].rolling(window=7, min_periods=1).mean().replace({0: np.nan})
        df['days_covered'] = (df['available_cash'] / daily_avg_outflow).replace([np.inf, -np.inf], np.nan).fillna(0)
        df['days_covered'] = df['days_covered'].clip(lower=0, upper=365)

        df['liquidity_risk'] = pd.cut(
            df['days_covered'],
            bins=[-1, 7, 15, 30, float('inf')],
            labels=['Critical', 'High', 'Moderate', 'Low']
        )

        # ensure string dtype
        df['liquidity_risk'] = df['liquidity_risk'].astype(object).fillna('Critical')

        return df

    # --- Category Analysis ---
    def categorized_analysis(self, inflows: pd.DataFrame, outflows: pd.DataFrame):
        """Analyze flows by category and currency (using base_amount). Returns dictionary of aggregated frames."""
        inflow_total = inflows['base_amount'].sum() if (inflows is not None and not inflows.empty) else 0.0
        outflow_total = outflows['base_amount'].sum() if (outflows is not None and not outflows.empty) else 0.0

        def agg(df, total):
            if df is None or df.empty:
                return pd.DataFrame(columns=['category', 'sum', 'mean', 'std', 'count', 'percentage'])
            agg_df = df.groupby('category')['base_amount'].agg(['sum', 'mean', 'std', 'count']).reset_index()
            agg_df['percentage'] = agg_df['sum'] / total * 100 if total > 0 else 0.0
            agg_df.rename(columns={'sum': 'sum'}, inplace=True)
            return agg_df

        inflow_by_category = agg(inflows, inflow_total)
        outflow_by_category = agg(outflows, outflow_total)

        def agg_currency(df, total):
            if df is None or df.empty:
                return pd.DataFrame(columns=['currency', 'sum', 'count', 'percentage'])
            agg_df = df.groupby('currency')['base_amount'].agg(['sum', 'count']).reset_index()
            agg_df['percentage'] = agg_df['sum'] / total * 100 if total > 0 else 0.0
            agg_df.rename(columns={'sum': 'sum'}, inplace=True)
            return agg_df

        inflow_by_currency = agg_currency(inflows, inflow_total)
        outflow_by_currency = agg_currency(outflows, outflow_total)

        return {
            'inflow_by_category': inflow_by_category,
            'outflow_by_category': outflow_by_category,
            'inflow_by_currency': inflow_by_currency,
            'outflow_by_currency': outflow_by_currency
        }

    # ------------------------------------------------------------------
    # --- ML/AI Enhancement 1: ARIMA Forecasting ---
    # ------------------------------------------------------------------
    def ml_forecast_net_flow(self, historical_net_flows: pd.DataFrame, forecast_days: int = 30, order=(5, 1, 0)):
        """
        Uses ARIMA to forecast net cash flow for the next forecast_days.
        Returns a DataFrame with date, net_forecast, lower_ci, upper_ci, cumulative_forecast.
        If the model can't be fit, returns an empty DataFrame.
        """
        if historical_net_flows is None or historical_net_flows.empty or len(historical_net_flows) < 10:
            return pd.DataFrame()

        # Prepare time series: set date index, resample to daily frequency and fill missing with 0 net (safe choice)
        ts = historical_net_flows.set_index('date')['net'].astype(float).sort_index()
        ts = ts.asfreq('D').fillna(0.0)

        try:
            model = ARIMA(ts, order=order)
            model_fit = model.fit()

            forecast_res = model_fit.get_forecast(steps=forecast_days)
            forecast_mean = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int()

            # Forecast dates
            start_date = ts.index[-1] + timedelta(days=1)
            end_date = start_date + timedelta(days=forecast_days - 1)
            forecast_dates = pd.date_range(start=start_date, end=end_date, freq='D')

            lower_col = conf_int.columns[0]
            upper_col = conf_int.columns[1]

            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'net_forecast': forecast_mean.values,
                'lower_ci': conf_int[lower_col].values,
                'upper_ci': conf_int[upper_col].values
            })

            # cumulative forecast starting from last known cumulative
            last_balance = historical_net_flows['cumulative'].iloc[-1]
            forecast_df['cumulative_forecast'] = last_balance + forecast_df['net_forecast'].cumsum()

            return forecast_df.reset_index(drop=True)

        except Exception as e:
            # log to stdout for debug (Streamlit will capture logs)
            print(f"ARIMA fit/predict error: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # --- ML/AI Enhancement 2: Cash-at-Risk (CaR) Liquidity Metric ---
    # ------------------------------------------------------------------
    def cash_at_risk_analysis(self, net_flows: pd.DataFrame, confidence_level: float = 0.95):
        """
        Calculates Cash-at-Risk (CaR) to assess the worst-case cash position using rolling volatility.
        Adds columns: rolling_std, cash_at_risk, worst_case_cumulative, worst_case_risk
        """
        if net_flows is None or net_flows.empty:
            return pd.DataFrame()

        df = net_flows.copy()
        daily_net = df['net'].astype(float)

        # rolling std - use 30-day window and fill initial NaNs with 0 (conservative / stable)
        df['rolling_std'] = daily_net.rolling(window=30, min_periods=1).std().fillna(0.0)

        # z-score for confidence
        z_score = norm.ppf(confidence_level)

        # cash at risk (1-day worst loss estimate)
        df['cash_at_risk'] = df['rolling_std'] * z_score
        df['cash_at_risk'] = df['cash_at_risk'].fillna(0.0)

        # worst-case cumulative if that day's worst loss materializes
        df['worst_case_cumulative'] = df['cumulative'] - df['cash_at_risk']

        # Better risk categorization using relative thresholds (7-day and 30-day outflow buffers)
        seven_day_avg_outflow = df['outflow'].rolling(window=7, min_periods=1).mean().fillna(0.0)
        thirty_day_avg_outflow = df['outflow'].rolling(window=30, min_periods=1).mean().fillna(0.0)

        def map_risk(row):
            if row['worst_case_cumulative'] < 0:
                return 'Critical'
            # if worst_case less than 7 days of avg outflow -> High
            if row['worst_case_cumulative'] < (seven_day_avg_outflow.loc[row.name] * 7):
                return 'High'
            # if worst_case less than 30 days of avg outflow -> Moderate
            if row['worst_case_cumulative'] < (thirty_day_avg_outflow.loc[row.name] * 30):
                return 'Moderate'
            return 'Low'

        # map row-wise; ensure index alignment
        df = df.reset_index(drop=True)
        df['worst_case_risk'] = [map_risk(df.loc[idx]) for idx in df.index]

        return df
