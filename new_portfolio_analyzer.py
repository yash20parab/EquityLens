import streamlit as st
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import pandas as pd
import feedparser
import yfinance as yf
import numpy as np


# Styles
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
        .stApp {
            background-color: #0a0a0a;
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }
        .stButton>button {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #333333;
            border-radius: 4px;
            transition: all 0.3s ease;
            font-family: 'Inter', sans-serif;
        }
        .stButton>button:hover {
            background-color: #333333;
            transform: scale(1.05);
        }
        .stTextInput>div>div>input {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #333333;
            border-radius: 4px;
            font-family: 'Inter', sans-serif;
        }
        .stSelectbox>div>div>select {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #333333;
            border-radius: 4px;
            font-family: 'Inter', sans-serif;
        }
        .stSlider>div>div>div>div {
            background-color: #1e1e1e;
        }
        .stMarkdown {
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }
        
        /* New Styles */
        .card {
            background: linear-gradient(145deg, #1e1e1e, #2a2a2a);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            font-family: 'Inter', sans-serif;
        }
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        .metric-card {
            text-align: center;
            padding: 1.5rem;
            background: linear-gradient(145deg, #1e1e1e, #2a2a2a);
            border-radius: 12px;
            margin: 0.5rem;
            transition: all 0.3s ease;
            font-family: 'Inter', sans-serif;
        }
        .metric-card:hover {
            transform: scale(1.02);
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            margin: 0.5rem 0;
            font-family: 'Inter', sans-serif;
        }
        .metric-label {
            color: #a0a0a0;
            font-size: 0.9rem;
            font-family: 'Inter', sans-serif;
        }
        @media (max-width: 768px) {
            .stColumn {
                flex: 1 1 100% !important;
                margin: 0.5rem 0;
            }
            .card {
                padding: 1rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Components
def create_navigation():
    return option_menu(
        menu_title=None,
        options=["Portfolio Analysis","News"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#1e1e1e"},
            "icon": {"color": "white", "font-size": "14px"}, 
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#333333"},
            "nav-link-selected": {"background-color": "#333333"},
        }
    )

def create_sidebar():
    st.sidebar.header("Portfolio Configuration", divider="gray")
    tickers = st.sidebar.text_input("Enter Indian stock tickers (comma separated, add .NS suffix)", "RELIANCE.NS,TCS.NS,HDFCBANK.NS")
    shares = st.sidebar.text_input("Enter number of shares (comma separated)", "10,20,30")
    return tickers, shares

def display_portfolio_metrics(portfolio_return, portfolio_std, sharpe_ratio):
    st.subheader("Portfolio Analysis Results", divider="gray")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">'
                   f'<div class="metric-value">{portfolio_return:.2%}</div>'
                   '<div class="metric-label">Portfolio Return</div>'
                   '</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">'
                   f'<div class="metric-value">{portfolio_std:.2%}</div>'
                   '<div class="metric-label">Portfolio Risk (Std Dev)</div>'
                   '</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">'
                   f'<div class="metric-value">{sharpe_ratio:.2f}</div>'
                   '<div class="metric-label">Sharpe Ratio</div>'
                   '</div>', unsafe_allow_html=True)

def display_stock_returns(tickers, annual_returns):
    st.subheader("Individual Stock Returns", divider="gray")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    for ticker, ret in zip(tickers, annual_returns):
        st.markdown(f'<div style="padding: 0.5rem 0;">{ticker}: <strong>{ret:.2%}</strong></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_portfolio_composition(tickers, current_values):
    st.subheader("Portfolio Composition", divider="gray")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    ax.pie(current_values, labels=tickers, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

def display_portfolio_performance(data, tickers, shares):
    portfolio_value = pd.Series(1.0, index=data[tickers[0]].index)
    for ticker, share_count in zip(tickers, shares):
        portfolio_value = portfolio_value * (1 + data[ticker]['DailyReturn'].fillna(0) * (share_count * data[ticker]['Adj Close'].iloc[-1]))

    st.subheader("Portfolio Performance", divider="gray")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    portfolio_value.plot(ax=ax)
    ax.set_ylabel("Portfolio Value")
    ax.set_title("Portfolio Growth Over Time")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

def display_news(ticker):
    st.subheader(f"Latest News for {ticker}", divider="gray")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    try:
        stock_name = ticker.replace('.NS', '')
        news_feed = feedparser.parse(f"https://news.google.com/rss/search?q={stock_name}+stock&hl=en-IN&gl=IN&ceid=IN:en")
        if news_feed.entries:
            for entry in news_feed.entries[:5]:
                st.markdown(f'<div style="margin-bottom: 1.5rem;">'
                          f'<div style="font-size: 1.1rem; font-weight: bold;">{entry.title}</div>'
                          f'<div style="color: #a0a0a0; font-size: 0.9rem;">{entry.source.title} - {entry.published}</div>'
                          f'<a href="{entry.link}" style="color: #4dabf7; text-decoration: none;">Read more →</a>'
                          '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color: #a0a0a0;">No recent news available for this stock.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div style="color: #ff6b6b;">Could not fetch news for {ticker}: {str(e)}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Utils
def process_inputs(tickers, shares):
    tickers = [ticker.strip() for ticker in tickers.split(",")]
    shares = [int(share) for share in shares.split(",")]
    return tickers, shares

def get_date_range():
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.Timedelta(days=365)
    return start_date, end_date

def fetch_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        df['DailyReturn'] = df['Adj Close'].pct_change()
        data[ticker] = df
    return data

def calculate_metrics(data, shares, tickers):
    annual_returns = []
    current_values = []
    for ticker, share_count in zip(tickers, shares):
        initial = data[ticker]['Adj Close'].iloc[0]
        final = data[ticker]['Adj Close'].iloc[-1]
        annual_returns.append((final / initial) - 1)
        current_values.append(final * share_count)
    
    total_value = sum(current_values)
    weights = np.array([value / total_value for value in current_values])
    
    returns_df = pd.DataFrame({ticker: data[ticker]['DailyReturn'].dropna() for ticker in tickers})
    returns_df = returns_df.dropna()
    cov_matrix = np.cov(returns_df.values.T)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    portfolio_return = np.dot(annual_returns, weights)
    sharpe_ratio = portfolio_return / portfolio_std
    
    return annual_returns, portfolio_return, portfolio_std, sharpe_ratio, weights, current_values

# Main App
def main():
    # Navigation


    selected = create_navigation()
    
    # Sidebar
    tickers, shares = create_sidebar()
    tickers, shares = process_inputs(tickers, shares)
    start_date, end_date = get_date_range()
    data = fetch_data(tickers, start_date, end_date)
    
    if selected == "Portfolio Analysis":
        annual_returns, portfolio_return, portfolio_std, sharpe_ratio, weights, current_values = calculate_metrics(data, shares, tickers)
        
        display_portfolio_metrics(portfolio_return, portfolio_std, sharpe_ratio)
        display_stock_returns(tickers, annual_returns)
        display_portfolio_composition(tickers, current_values)
        display_portfolio_performance(data, tickers, shares)
    elif selected == "News":
        for ticker in tickers:
            display_news(ticker)

if __name__ == "__main__":
    main()
