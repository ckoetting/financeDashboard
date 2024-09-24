import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Function to map ISIN to Ticker Symbol
def isin_to_ticker(isin):
    isin_ticker_map = {
        # Long-Term Assets
        'IE00B4L5Y983': 'IWDA.AS',  # iShares Core MSCI World UCITS ETF USD (Acc)
        'IE00BKM4GZ66': 'EIMI.L',   # iShares Core MSCI EM IMI UCITS ETF
        'US81369Y8030': 'XLF',      # Financial Select Sector SPDR Fund (Sector ETF)
        # Add more mappings as needed
    }
    return isin_ticker_map.get(isin, None)

# Pre-selected ISIN codes
default_isins = {
    # Long-Term Investments
    "MSCI Global ETF ISIN": 'IE00B4L5Y983',
    "Developing Countries ETF ISIN": 'IE00BKM4GZ66',
    "Sector-Specific ETF ISIN": 'US81369Y8030',
}

st.title("Investment Portfolio Dashboard")

st.write("---")

# Investment Inputs
st.header("Investment Inputs")

# Create columns for amounts and horizons
amount_col, horizon_col = st.columns(2)

with amount_col:
    st.subheader("Investment Amounts")

    # Short-Medium Term Investment
    short_term_investment = st.number_input(
        "Short-Medium Term Investment Amount (€)",
        min_value=0,
        value=35000,
        step=1000,
        format="%d"
    )

    # Long-Term Investment
    long_term_investment = st.number_input(
        "Long-Term Investment Amount (€)",
        min_value=0,
        value=10000,
        step=1000,
        format="%d"
    )

with horizon_col:
    st.subheader("Investment Horizons")

    # Short-Medium Term Horizon (Months)
    short_term_horizon = st.number_input(
        "Short-Medium Term Investment Horizon (Months)",
        min_value=1,
        value=6,
        step=1,
        format="%d"
    )

    # Long-Term Horizon (Years)
    long_term_horizon = st.number_input(
        "Long-Term Investment Horizon (Years)",
        min_value=1,
        value=10,
        step=1,
        format="%d"
    )

st.write("---")

# Inflation Rate Input
st.header("Assumed Inflation Rate")
inflation_rate = st.number_input(
    "Annual Inflation Rate (%)",
    min_value=0.0,
    value=2.0,
    step=0.1,
    format="%.1f",
    help="Enter the expected average annual inflation rate."
) / 100

st.write("---")

# Short-Term Investment Rates
st.header("Short-Term Investment Rates")
savings_rate = st.number_input(
    "High-Yield Savings Account Annual Rate (%)",
    min_value=0.0,
    value=1.5,
    step=0.1,
    format="%.1f",
    help="Enter the annual interest rate for the high-yield savings account."
) / 100

gov_bonds_rate = st.number_input(
    "Government Bonds Annual Rate (%)",
    min_value=0.0,
    value=2.0,
    step=0.1,
    format="%.1f",
    help="Enter the annual interest rate for government bonds."
) / 100

corp_bonds_rate = st.number_input(
    "Corporate Bonds Annual Rate (%)",
    min_value=0.0,
    value=3.0,
    step=0.1,
    format="%.1f",
    help="Enter the annual interest rate for corporate bonds."
) / 100

st.write("---")

# Asset Allocations
st.header("Asset Allocations")

# Short-Medium Term Allocations
st.subheader("Short-Medium Term Asset Allocation (%)")
short_alloc_col1, short_alloc_col2, short_alloc_col3 = st.columns(3)

with short_alloc_col1:
    savings_pct = st.slider(
        "High-Yield Savings Account (%)",
        min_value=0,
        max_value=100,
        value=34,
        step=1
    )

with short_alloc_col2:
    gov_bonds_pct = st.slider(
        "Government Bonds (%)",
        min_value=0,
        max_value=100,
        value=33,
        step=1
    )

with short_alloc_col3:
    corp_bonds_pct = st.slider(
        "Corporate Bonds (%)",
        min_value=0,
        max_value=100,
        value=33,
        step=1
    )

# Check allocation sum
short_term_total_alloc = savings_pct + gov_bonds_pct + corp_bonds_pct
if short_term_total_alloc != 100:
    st.error("Short-Medium Term allocations must sum up to 100%.")

# Long-Term Allocations
st.subheader("Long-Term Asset Allocation (%)")
long_alloc_col1, long_alloc_col2, long_alloc_col3 = st.columns(3)

with long_alloc_col1:
    msci_global_pct = st.slider(
        "MSCI Global ETF (%)",
        min_value=0,
        max_value=100,
        value=34,
        step=1
    )

with long_alloc_col2:
    developing_countries_pct = st.slider(
        "Developing Countries ETF (%)",
        min_value=0,
        max_value=100,
        value=33,
        step=1
    )

with long_alloc_col3:
    sector_specific_pct = st.slider(
        "Sector-Specific ETF (%)",
        min_value=0,
        max_value=100,
        value=33,
        step=1
    )

# Check allocation sum
long_term_total_alloc = msci_global_pct + developing_countries_pct + sector_specific_pct
if long_term_total_alloc != 100:
    st.error("Long-Term allocations must sum up to 100%.")

# Long-Term ETF Selection
st.subheader("Long-Term ETFs Selection")
msci_global_isin = st.text_input(
    "MSCI Global ETF ISIN",
    value=default_isins["MSCI Global ETF ISIN"]
)
developing_countries_isin = st.text_input(
    "Developing Countries ETF ISIN",
    value=default_isins["Developing Countries ETF ISIN"]
)
sector_specific_isin = st.text_input(
    "Sector-Specific ETF ISIN",
    value=default_isins["Sector-Specific ETF ISIN"]
)

msci_global_ticker = isin_to_ticker(msci_global_isin)
developing_countries_ticker = isin_to_ticker(developing_countries_isin)
sector_specific_ticker = isin_to_ticker(sector_specific_isin)

st.write("---")

# Convert annual rates to monthly rates for short-term investments
rates = {
    "High-Yield Savings Account": savings_rate,
    "Government Bonds": gov_bonds_rate,
    "Corporate Bonds": corp_bonds_rate
}

monthly_rates = {}
for asset, annual_rate in rates.items():
    monthly_rate = (1 + annual_rate) ** (1/12) - 1
    monthly_rates[asset] = monthly_rate

# Collect all ETFs
etf_tickers = {
    "MSCI Global ETF": msci_global_ticker,
    "Developing Countries ETF": developing_countries_ticker,
    "Sector-Specific ETF": sector_specific_ticker
}

# Remove None tickers
etf_tickers = {k: v for k, v in etf_tickers.items() if v is not None}

# Fetch historical returns for ETFs
@st.cache_data
def get_etf_annual_stats(ticker):
    data = yf.download(ticker, period="10y")
    if data.empty:
        return None, None
    data['Daily Return'] = data['Adj Close'].pct_change()
    annual_returns = data['Daily Return'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
    mean_return = annual_returns.mean()
    std_return = annual_returns.std()
    return mean_return, std_return

# Fetch and store annual mean and std returns
etf_stats = {}
for etf_name, ticker in etf_tickers.items():
    mean_return, std_return = get_etf_annual_stats(ticker)
    if mean_return is not None:
        etf_stats[etf_name] = {'mean': mean_return, 'std': std_return}
    else:
        st.error(f"Data for {etf_name} ({ticker}) is unavailable.")

# Perform Investment Calculations

# Short-Medium Term Calculations
savings_alloc = savings_pct / 100
gov_bonds_alloc = gov_bonds_pct / 100
corp_bonds_alloc = corp_bonds_pct / 100

# Calculate future values using monthly compounding
savings_future = short_term_investment * savings_alloc * (1 + monthly_rates["High-Yield Savings Account"]) ** short_term_horizon
gov_bonds_future = short_term_investment * gov_bonds_alloc * (1 + monthly_rates["Government Bonds"]) ** short_term_horizon
corp_bonds_future = short_term_investment * corp_bonds_alloc * (1 + monthly_rates["Corporate Bonds"]) ** short_term_horizon

short_term_total_future = savings_future + gov_bonds_future + corp_bonds_future
short_term_total_gain = short_term_total_future - short_term_investment

# Adjust for inflation
inflation_monthly_rate = (1 + inflation_rate) ** (1/12) - 1
inflation_adjustment = (1 + inflation_monthly_rate) ** short_term_horizon
short_term_total_future_real = short_term_total_future / inflation_adjustment

# Long-Term Calculations
msci_global_alloc = msci_global_pct / 100
developing_countries_alloc = developing_countries_pct / 100
sector_specific_alloc = sector_specific_pct / 100

# Monte Carlo Simulations for Long-Term Investments
if all(etf in etf_stats for etf in ["MSCI Global ETF", "Developing Countries ETF", "Sector-Specific ETF"]):
    num_simulations = 1000
    projection_years = np.arange(0, int(long_term_horizon) + 1)
    simulation_results = pd.DataFrame({'Year': projection_years})

    for sim in range(num_simulations):
        total_portfolio = np.zeros(len(projection_years))
        for etf_name in ["MSCI Global ETF", "Developing Countries ETF", "Sector-Specific ETF"]:
            stats = etf_stats[etf_name]
            alloc = 0
            if etf_name == "MSCI Global ETF":
                alloc = msci_global_alloc
            elif etf_name == "Developing Countries ETF":
                alloc = developing_countries_alloc
            elif etf_name == "Sector-Specific ETF":
                alloc = sector_specific_alloc

            initial_investment = long_term_investment * alloc
            # Simulate annual returns
            random_returns = np.random.normal(stats['mean'], stats['std'], int(long_term_horizon))
            cumulative_returns = np.cumprod(1 + random_returns)
            # Include initial investment year
            cumulative_returns = np.insert(cumulative_returns, 0, 1.0)
            future_values = initial_investment * cumulative_returns
            total_portfolio += future_values

        simulation_results[f"Simulation_{sim}"] = total_portfolio

    # Adjust for inflation
    inflation_adjustment_long = (1 + inflation_rate) ** projection_years
    for col in simulation_results.columns[1:]:
        simulation_results[col] = simulation_results[col] / inflation_adjustment_long

    # Calculate percentiles
    percentiles = [5, 25, 50, 75, 95]
    percentile_values = simulation_results.iloc[:, 1:].quantile(q=np.array(percentiles)/100, axis=1)
    percentile_df = pd.DataFrame(percentile_values.values.T, columns=[f"{p}th Percentile" for p in percentiles])
    percentile_df['Year'] = projection_years

    # Value at Risk (VaR) Calculation at 95% confidence level
    final_values = simulation_results.iloc[-1, 1:]
    VaR_95_real = np.percentile(final_values, 5)
    expected_final_value_real = np.mean(final_values)
else:
    st.error("Insufficient data for long-term ETFs.")

# Display Results
st.header("Short-Medium Term Investment Results")

short_term_breakdown = pd.DataFrame({
    'Investment Type': ['High-Yield Savings Account', 'Government Bonds', 'Corporate Bonds'],
    'Allocation (€)': [
        short_term_investment * savings_alloc,
        short_term_investment * gov_bonds_alloc,
        short_term_investment * corp_bonds_alloc
    ],
    'Future Value (€)': [savings_future, gov_bonds_future, corp_bonds_future],
    'Inflation Adjusted (€)': [
        (savings_future / inflation_adjustment),
        (gov_bonds_future / inflation_adjustment),
        (corp_bonds_future / inflation_adjustment)
    ]
})

st.write(f"**Total Future Value (Nominal):** €{short_term_total_future:,.2f}")
st.write(f"**Total Future Value (Inflation Adjusted):** €{short_term_total_future_real:,.2f}")
st.write(f"**Total Gain (Nominal):** €{short_term_total_gain:,.2f}")

# Plotly Pie Chart
fig_short_term = px.pie(
    short_term_breakdown,
    values='Future Value (€)',
    names='Investment Type',
    title='Short-Medium Term Investment Distribution',
    hole=0.4
)
st.plotly_chart(fig_short_term, use_container_width=True)

# Short-Term Investment Performance Over Time
st.header("Short-Medium Term Investment Growth Over Time")

# Create a range of months
time_horizon = np.arange(0, short_term_horizon + 1)  # From month 0 to the investment horizon

# Calculate balances for each month
savings_balance = short_term_investment * savings_alloc * (1 + monthly_rates["High-Yield Savings Account"]) ** time_horizon
gov_bonds_balance = short_term_investment * gov_bonds_alloc * (1 + monthly_rates["Government Bonds"]) ** time_horizon
corp_bonds_balance = short_term_investment * corp_bonds_alloc * (1 + monthly_rates["Corporate Bonds"]) ** time_horizon

# Adjust balances for inflation
inflation_adjustments = (1 + inflation_monthly_rate) ** time_horizon
savings_balance_real = savings_balance / inflation_adjustments
gov_bonds_balance_real = gov_bonds_balance / inflation_adjustments
corp_bonds_balance_real = corp_bonds_balance / inflation_adjustments
total_balance_real = savings_balance_real + gov_bonds_balance_real + corp_bonds_balance_real

# Create a DataFrame for plotting
growth_df = pd.DataFrame({
    'Month': time_horizon,
    'High-Yield Savings Account': savings_balance_real,
    'Government Bonds': gov_bonds_balance_real,
    'Corporate Bonds': corp_bonds_balance_real,
    'Total Portfolio': total_balance_real
})

# Melt the DataFrame to a long format for Plotly
growth_df_melted = growth_df.melt(id_vars='Month', var_name='Investment Type', value_name='Balance (€)')

# Plot the investment growth over time
fig_growth = px.line(
    growth_df_melted,
    x='Month',
    y='Balance (€)',
    color='Investment Type',
    title='Short-Medium Term Investment Growth Over Time (Inflation Adjusted)',
    markers=True
)
st.plotly_chart(fig_growth, use_container_width=True)

# Long-Term Results
st.header("Long-Term Investment Results")

if 'simulation_results' in locals():
    st.write(f"**Expected Final Portfolio Value (Inflation Adjusted):** €{expected_final_value_real:,.2f}")
    st.write(f"**5th Percentile Final Portfolio Value (Inflation Adjusted VaR at 95% Confidence Level):** €{VaR_95_real:,.2f}")

    # Plot the projections with shaded areas
    fig = go.Figure()

    # Add shaded area between 5th and 95th percentiles
    fig.add_trace(go.Scatter(
        x=percentile_df['Year'],
        y=percentile_df['5th Percentile'],
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=percentile_df['Year'],
        y=percentile_df['95th Percentile'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(173,216,230,0.2)',  # Light blue
        line=dict(color='rgba(0,0,0,0)'),
        name='5th-95th Percentile Range',
        hoverinfo='skip'
    ))

    # Add shaded area between 25th and 75th percentiles
    fig.add_trace(go.Scatter(
        x=percentile_df['Year'],
        y=percentile_df['25th Percentile'],
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=percentile_df['Year'],
        y=percentile_df['75th Percentile'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(173,216,230,0.4)',  # Darker blue
        line=dict(color='rgba(0,0,0,0)'),
        name='25th-75th Percentile Range',
        hoverinfo='skip'
    ))

    # Add percentile lines with distinct colors
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    for idx, percentile in enumerate(percentiles):
        fig.add_trace(go.Scatter(
            x=percentile_df['Year'],
            y=percentile_df[f"{percentile}th Percentile"],
            mode='lines+markers',
            name=f'{percentile}th Percentile',
            line=dict(color=colors[idx], width=2),
            marker=dict(size=6),
            text=[f"Year {int(year)}: €{value:,.2f}" for year, value in zip(percentile_df['Year'], percentile_df[f"{percentile}th Percentile"])],
            hovertemplate='%{text}<extra></extra>'
        ))

    # Adjust legend placement and styling
    fig.update_layout(
        title='Projected Long-Term Investment Growth (Inflation Adjusted)',
        xaxis_title='Year',
        yaxis_title='Portfolio Value (€)',
        hovermode='closest',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,  # Move legend below the chart
            xanchor='center',
            x=0.5,
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0.5)'  # Semi-transparent background
        ),
        margin=dict(l=50, r=50, t=80, b=120)  # Adjust margins to accommodate legend
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Long-term investment results are not available due to insufficient data.")

st.write("""
## Understanding Monte Carlo Simulations and Value at Risk (VaR)

### Monte Carlo Simulations:

Monte Carlo simulations are a mathematical technique used to estimate the possible outcomes of an uncertain event. In the context of investments:

- **Purpose**: To model the potential future performance of an investment portfolio by simulating a large number of possible scenarios.
- **How It Works**:
  - **Random Sampling**: The simulations generate random returns based on the historical mean and volatility (standard deviation) of each asset.
  - **Multiple Paths**: Each simulation represents a different possible future path your investment could take.
  - **Aggregation**: By running many simulations (e.g., 1,000), we can observe a range of possible outcomes and their probabilities.

### Interpreting the Chart:

- **Shaded Areas**:
  - The **light blue shaded area** represents the range between the **5th and 95th percentiles**, indicating where 90% of the simulated outcomes fall.
  - The **darker blue shaded area** represents the range between the **25th and 75th percentiles**, showing where the middle 50% of outcomes lie.
- **Percentile Lines**:
  - **5th Percentile (Red Line)**: A **pessimistic** outcome. There's a 95% chance your investment will perform better than this line.
  - **25th Percentile (Orange Line)**: A **conservative** estimate, showing lower expected returns.
  - **50th Percentile (Green Line)**: The **median** outcome. There's a 50% chance your investment will perform better or worse than this line.
  - **75th Percentile (Blue Line)**: An **optimistic** estimate, indicating higher potential returns.
  - **95th Percentile (Purple Line)**: A **very optimistic** outcome. There's a 5% chance your investment will perform better than this line.

### Value at Risk (VaR):

- **Definition**: VaR is a statistical measure that estimates the potential loss in value of an investment portfolio over a specified time period, given normal market conditions, at a certain confidence level.
- **In This Dashboard**:
  - **VaR at 95% Confidence Level**: Indicates that there is a 5% chance your investment could be worth €X or less at the end of the investment horizon.
  - **Interpretation**: It helps you understand the potential downside risk of your investment.

### Inflation Adjustment:

- **Why Adjust for Inflation**:
  - **Purchasing Power**: Inflation erodes the purchasing power of money over time.
  - **Real vs. Nominal Values**: Adjusting for inflation gives you the **real** value of your investment, reflecting what you can actually buy in the future.
- **In the Simulations**:
  - All projected values are adjusted for the expected annual inflation rate you've input, providing a more accurate picture of future value.

### Key Takeaways:

- **Uncertainty in Investments**: The future performance of investments is uncertain, and Monte Carlo simulations help illustrate that uncertainty.
- **Risk Assessment**: Understanding the range of possible outcomes and the associated risks can help in making informed investment decisions.
- **Long-Term Perspective**: Investing over a longer horizon generally increases the likelihood of achieving better returns, but it's important to be aware of potential risks along the way.

**Disclaimer**: The projections and simulations are based on historical data and assumptions. They are for illustrative purposes only and do not guarantee future results. Always consider consulting with a financial advisor for personalized advice.

""")
