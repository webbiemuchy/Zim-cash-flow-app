import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class ZimbabweCashFlowModel:
    def __init__(self, base_currency='USD', initial_balance=25000, zig_volatility=0.02, zar_volatility=0.01, business_scale=1.0):
        self.base_currency = base_currency
        self.initial_balance = initial_balance
        self.zig_volatility = zig_volatility
        self.zar_volatility = zar_volatility
        self.business_scale = business_scale
        
        # Updated base rates to target your specified ranges
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

    def _get_current_zig_rate(self):
        base_rate = 26.8337
        volatility = np.random.normal(0, self.zig_volatility)
        return base_rate * (1 + volatility)

    def _get_current_zar_rate(self):
        base_rate = 18.25
        volatility = np.random.normal(0, self.zar_volatility)
        return base_rate * (1 + volatility)
    
    def _update_exchange_rates(self, date):
        zig_trend = np.sin(date.day / 10) * 0.02
        zig_random = np.random.normal(0, self.zig_volatility / 2)
        zar_trend = np.sin(date.day / 15) * 0.01
        zar_random = np.random.normal(0, self.zar_volatility / 2)
        
        self.exchange_rates['ZiG'] *= (1 + zig_trend + zig_random)
        self.exchange_rates['ZAR'] *= (1 + zar_trend + zar_random)
        
        return self.exchange_rates

    def simulate_transactions(self, days=30, categorized=True):
        """Simulate cash flows with realistic Zimbabwe market conditions"""
        start_date = datetime.now()
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Initialize DataFrames to store all transaction details
        inflows = pd.DataFrame(index=range(len(dates) * len(self.transaction_categories['inflows'])))
        outflows = pd.DataFrame(index=range(len(dates) * len(self.transaction_categories['outflows'])))
        
        row_in, row_out = 0, 0
        
        # Base amounts for different business sizes, scaled by business_scale parameter
        base_inflow = 12000 * self.business_scale
        base_outflow = 10000 * self.business_scale
        
        for day_idx, day in enumerate(dates):
            # Apply seasonal factors
            month_factor = self.seasonality['monthly'][day.month]
            weekday_factor = self.seasonality['weekly'][day.weekday()]
            seasonal_factor = month_factor * weekday_factor
            
            # Update exchange rates for the day
            daily_rates = self._update_exchange_rates(day)
            
            # Generate inflows by category
            for category, params in self.transaction_categories['inflows'].items():
                # Calculate the base amount for this category
                category_base = base_inflow * params['weight'] * seasonal_factor
                
                # Add volatility specific to this category
                category_amount = category_base * (1 + np.random.normal(0, params['volatility']))
                
                # Distribute across currencies
                for i, currency in enumerate(params['currencies']):
                    if np.random.random() < 0.7:  # 70% chance of having a transaction in this currency
                        amount = category_amount * params['distribution'][i]
                        
                        # Add some randomness to the amount
                        amount *= np.random.uniform(0.9, 1.1)
                        
                        # Convert to the actual currency amount
                        currency_amount = amount * daily_rates[currency] if currency != self.base_currency else amount
                        
                        # Record the transaction
                        inflows.loc[row_in, 'date'] = day
                        inflows.loc[row_in, 'category'] = category
                        inflows.loc[row_in, 'currency'] = currency
                        inflows.loc[row_in, 'amount'] = max(0, currency_amount)  # Ensure positive
                        inflows.loc[row_in, f'{currency}_rate'] = daily_rates[currency]
                        
                        # Add potential payment delay
                        delay_key = f"{'local' if currency == 'ZiG' else 'regional' if currency == 'ZAR' else 'international'}_{currency}"
                        delay_params = self.payment_delays.get(delay_key, self.payment_delays['local_ZiG'])
                        
                        if np.random.random() < delay_params['probability']:
                            delay_days = int(max(0, np.random.normal(delay_params['mean_days'], delay_params['std_days'])))
                            inflows.loc[row_in, 'expected_date'] = day + timedelta(days=delay_days)
                        else:
                            inflows.loc[row_in, 'expected_date'] = day
                            
                        row_in += 1
            
            # Generate outflows by category
            for category, params in self.transaction_categories['outflows'].items():
                # Calculate the base amount for this category
                category_base = base_outflow * params['weight'] * seasonal_factor
                
                # Add volatility specific to this category
                category_amount = category_base * (1 + np.random.normal(0, params['volatility']))
                
                # Distribute across currencies
                for i, currency in enumerate(params['currencies']):
                    if np.random.random() < 0.8:  # 80% chance of having an expense in this currency
                        amount = category_amount * params['distribution'][i]
                        
                        # Add some randomness to the amount
                        amount *= np.random.uniform(0.95, 1.05)
                        
                        # Convert to the actual currency amount
                        currency_amount = amount * daily_rates[currency] if currency != self.base_currency else amount
                        
                        # Record the transaction
                        outflows.loc[row_out, 'date'] = day
                        outflows.loc[row_out, 'category'] = category
                        outflows.loc[row_out, 'currency'] = currency
                        outflows.loc[row_out, 'amount'] = max(0, currency_amount)  # Ensure positive
                        outflows.loc[row_out, f'{currency}_rate'] = daily_rates[currency]
                        
                        # Add mandatory payments on certain days (e.g., salaries on the 25th)
                        if category == 'salaries' and day.day == 25:
                            outflows.loc[row_out, 'amount'] *= 1.2  # Higher salary payments on payday
                        
                        # Add quarterly tax payments
                        if category == 'tax_payments' and day.day == 15 and day.month in [3, 6, 9, 12]:
                            outflows.loc[row_out, 'amount'] *= 3.0  # Quarterly tax is 3x regular provisions
                            
                        row_out += 1
        
        # Clean up and remove any NaN rows
        inflows = inflows.dropna(subset=['date', 'amount'])
        outflows = outflows.dropna(subset=['date', 'amount'])
        
        return inflows, outflows

    

    def calculate_net_flows(self, inflows, outflows):
        """Calculate net flows converting all currencies to base_currency"""
        # Create a date range for all days in the simulation
        if len(inflows) == 0 or len(outflows) == 0:
            return pd.DataFrame()
            
        start_date = min(inflows['date'].min(), outflows['date'].min())
        end_date = max(inflows['date'].max(), outflows['date'].max())
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create DataFrame with all dates
        net_flows = pd.DataFrame(date_range, columns=['date'])
        
        # Convert all amounts to base currency and group by date
        inflows_by_date = inflows.copy()
        outflows_by_date = outflows.copy()
        
        # Apply currency conversion
        for idx, row in inflows_by_date.iterrows():
            currency = row['currency']
            rate = row[f'{currency}_rate']
            inflows_by_date.loc[idx, 'base_amount'] = row['amount'] / rate
            
        for idx, row in outflows_by_date.iterrows():
            currency = row['currency']
            rate = row[f'{currency}_rate']
            outflows_by_date.loc[idx, 'base_amount'] = row['amount'] / rate
        
        # Group by date (and optionally by category)
        in_grouped = inflows_by_date.groupby('date')['base_amount'].sum().reset_index()
        out_grouped = outflows_by_date.groupby('date')['base_amount'].sum().reset_index()
        
        # Merge with the date range
        net_flows = pd.merge(net_flows, in_grouped, on='date', how='left').fillna(0)
        net_flows = pd.merge(net_flows, out_grouped, on='date', how='left').fillna(0)
        
        # Rename columns
        net_flows.rename(columns={'base_amount_x': 'inflow', 'base_amount_y': 'outflow'}, inplace=True)
        
        # Calculate net flow
        net_flows['net'] = net_flows['inflow'] - net_flows['outflow']
        
        # Add initial balance and calculate cumulative balance
        net_flows['cumulative'] = self.initial_balance + net_flows['net'].cumsum()
        
        return net_flows
    
    def analyze_liquidity(self, net_flows):
        """Enhanced liquidity analysis with Zimbabwe-specific factors"""
        if len(net_flows) == 0:
            return pd.DataFrame()
            
        # Calculate various liquidity metrics
        
        # 1. Available cash - what can actually be accessed immediately
        # In Zimbabwe context, not all reported cash is actually available due to:
        # - Bank withdrawal limits
        # - Foreign currency access restrictions
        # - Bank system outages
        availability_factor = 0.5884  # 58.84% of reported cash is actually accessible
        net_flows['available_cash'] = net_flows['cumulative'] * availability_factor
        
        # 2. Liquidity gap - when available cash is negative
        net_flows['liquidity_gap'] = net_flows['available_cash'].apply(lambda x: max(0, -x))
        
        # 3. Cash buffer - days of expenses covered by current balance
        daily_avg_outflow = net_flows['outflow'].rolling(window=7, min_periods=1).mean()
        net_flows['days_covered'] = net_flows['available_cash'] / daily_avg_outflow
        net_flows['days_covered'] = net_flows['days_covered'].fillna(0).clip(0, 90)  # Cap at 90 days
        
        # 4. Liquidity risk indicators
        net_flows['liquidity_risk'] = pd.cut(
            net_flows['days_covered'],
            bins=[-1, 7, 15, 30, float('inf')],
            labels=['Critical', 'High', 'Moderate', 'Low']
        )
        
        # 5. Weekly volatility measurement
        net_flows['volatility'] = net_flows['net'].rolling(window=7, min_periods=1).std()
        
        return net_flows
    
    def categorized_analysis(self, inflows, outflows):
        """Analyze flows by category to identify trends and dependencies"""
        # Group inflows by category and calculate stats
        inflow_by_category = inflows.groupby('category').agg({
            'base_amount': ['sum', 'mean', 'std', 'count']
        })
        
        # Calculate percentage distribution
        inflow_total = inflow_by_category[('base_amount', 'sum')].sum()
        inflow_by_category['percentage'] = inflow_by_category[('base_amount', 'sum')] / inflow_total * 100
        
        # Do the same for outflows
        outflow_by_category = outflows.groupby('category').agg({
            'base_amount': ['sum', 'mean', 'std', 'count']
        })
        
        outflow_total = outflow_by_category[('base_amount', 'sum')].sum()
        outflow_by_category['percentage'] = outflow_by_category[('base_amount', 'sum')] / outflow_total * 100
        
        # Currency distribution
        inflow_by_currency = inflows.groupby('currency').agg({
            'base_amount': ['sum', 'count']
        })
        inflow_by_currency['percentage'] = inflow_by_currency[('base_amount', 'sum')] / inflow_total * 100
        
        outflow_by_currency = outflows.groupby('currency').agg({
            'base_amount': ['sum', 'count']
        })
        outflow_by_currency['percentage'] = outflow_by_currency[('base_amount', 'sum')] / outflow_total * 100
        
        return {
            'inflow_by_category': inflow_by_category,
            'outflow_by_category': outflow_by_category,
            'inflow_by_currency': inflow_by_currency,
            'outflow_by_currency': outflow_by_currency
        }


    def forecast_exchange_rates(self, days=30):
        start_date = datetime.now()
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        forecast_data = {'date': [], 'ZiG_rate': [], 'ZAR_rate': []}
        
        zig_rate = self.exchange_rates['ZiG']
        zar_rate = self.exchange_rates['ZAR']
        
        for date in dates:
            # Smooth cyclical component to keep within desired range
            zig_trend = np.sin(date.day / 7) * 0.01
            zig_random = np.random.normal(0, self.zig_volatility / 2)
            zig_rate = np.clip(zig_rate * (1 + zig_trend + zig_random), 31, 38)
            
            zar_trend = np.sin(date.day / 10) * 0.005
            zar_random = np.random.normal(0, self.zar_volatility / 2)
            zar_rate = np.clip(zar_rate * (1 + zar_trend + zar_random), 18, 22)
            
            forecast_data['date'].append(date)
            forecast_data['ZiG_rate'].append(zig_rate)
            forecast_data['ZAR_rate'].append(zar_rate)
        
        return pd.DataFrame(forecast_data)

    def plot_exchange_rate_forecast(self, forecast_df):
        plt.figure(figsize=(12, 6))
        
        plt.plot(forecast_df['date'], forecast_df['ZiG_rate'], label='ZiG Rate', color='blue')
        plt.plot(forecast_df['date'], forecast_df['ZAR_rate'], label='ZAR Rate', color='green')
        
        plt.title('Zimbabwe Exchange Rate Forecast')
        plt.xlabel('Date')
        plt.ylabel('Exchange Rate')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Run Forecast and Plot
model = ZimbabweCashFlowModel()
forecast_df = model.forecast_exchange_rates(30)
model.plot_exchange_rate_forecast(forecast_df)
