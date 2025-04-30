
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Sidebar Input ---
st.title("📉 Portfolio Value at Risk (VaR) Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload CSV with historical prices", type=["csv"])
confidence = st.sidebar.slider("Confidence Level", min_value=90, max_value=99, value=95)
simulations = st.sidebar.number_input("Monte Carlo Simulations", value=10000, step=1000)

# --- Main Logic ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Basic cleaning for known format
    try:
        close_prices = df[['Date', 'Close_AAPL', 'Close_MSFT', 'Close_GOOGL', 'Close_AMZN', 'Close_NVDA']].copy()
    except KeyError:
        st.error("Dataset must include: Close_AAPL, Close_MSFT, Close_GOOGL, Close_AMZN, Close_NVDA")
        st.stop()

    close_prices['Date'] = pd.to_datetime(close_prices['Date'])
    close_prices.set_index('Date', inplace=True)
    close_prices.sort_index(inplace=True)
    close_prices.columns = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    weights = np.array([1/5] * 5)
    port_returns = log_returns.dot(weights)

    # --- VaR Calculations ---
    var_level = 100 - confidence
    hist_var = np.percentile(port_returns, var_level)

    z_score = norm.ppf(1 - (var_level/100))
    param_var = z_score * port_returns.std()

    np.random.seed(42)
    mean_r, std_r = port_returns.mean(), port_returns.std()
    sim_returns = np.random.normal(mean_r, std_r, size=simulations)
    mc_var = np.percentile(sim_returns, var_level)

    cvar = port_returns[port_returns <= hist_var].mean()

    # --- Results ---
    st.subheader("1-Day VaR Estimates")
    st.metric("Historical VaR", f"{hist_var:.2%}")
    st.metric("Parametric VaR", f"{param_var:.2%}")
    st.metric("Monte Carlo VaR", f"{mc_var:.2%}")
    st.metric("Conditional VaR (CVaR)", f"{cvar:.2%}")

    # --- Histogram Plot ---
    st.subheader("Distribution of Portfolio Returns")
    fig, ax = plt.subplots()
    ax.hist(port_returns, bins=100, color='skyblue', edgecolor='black')
    ax.axvline(hist_var, color='red', linestyle='--', label=f'Historical VaR')
    ax.axvline(param_var, color='green', linestyle='--', label=f'Parametric VaR')
    ax.axvline(mc_var, color='purple', linestyle='--', label=f'Monte Carlo VaR')
    ax.set_title('Portfolio Daily Returns with VaR Thresholds')
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

    # --- Stress Test ---
    st.subheader("Stress Test: Market Shock Scenarios")
    for drop in [-0.05, -0.10, -0.15]:
        stressed = np.dot(weights, [drop]*len(weights))
        st.write(f"If all assets drop {int(-drop*100)}%, portfolio drops: {stressed:.2%}")
