import streamlit as st
import yfinance as yf
import requests
from requests.exceptions import RequestException
import time
import pandas as pd
import feedparser
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
import statsmodels.api as sm
import plotly.express as px
import matplotlib.pyplot as plt
import google.generativeai as genai  # Import Google Gemini for AI-based summary
from PIL import Image
from streamlit_option_menu import option_menu


# Streamlit app layout
st.set_page_config(page_title="EquityLens", layout="wide")
st.title("EquityLens")

# Navigation

selected = option_menu(
    menu_title=None,
    options=["Home","Market Status","Portfolio Analysis and News"],
    default_index=0,
    orientation="horizontal",

)

if selected == "Home":
    # Home Page Content
    
    
    # Header section
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        
        st.title("Stock Fundamental & Technical values")

    st.markdown("---")


    def get_symbol(symbol):
        return f"{symbol}"

    def format_large_number(number):
        if number >= 1e9:
            return f"{number/1e9:.2f} b"
        elif number >= 1e6:
            return f"{number/1e6:.2f} m"
        else:
            return f"{number:,.0f}"

    def analyze_stock(stock_symbol):
        try:
            stock = yf.Ticker(get_symbol(stock_symbol))
            # Daily data for general metrics
            daily_data = stock.history(period="1y", interval="1d")
            info = stock.info

            st.markdown("---")
            
            if not daily_data.empty:
                # header card
                st.success(f"📊 {stock_symbol} - {info.get('longName', stock_symbol)} Analysis report")
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                last_price = daily_data['Close'][-1]
                previous_price = daily_data['Close'][-2]
                change = ((last_price - previous_price) / previous_price) * 100
                
                with col1:
                    st.metric("LAST CLOSE", f"{last_price:.2f}", f"{change:.2f}%")
                with col2:
                    if 'marketCap' in info:
                        st.metric("MARKET CAP", format_large_number(info['marketCap']))
                with col3:
                    if 'volume' in info:
                        st.metric("VOLUME", format_large_number(daily_data['Volume'][-1]))
                with col4:
                    if 'fiftyTwoWeekHigh' in info:
                        st.metric("52 WEEK HIGH", f"{info['fiftyTwoWeekHigh']:.2f}")

                st.markdown("---")
                
                # FUNDAMENTAL ANALYSIS METRICS
                st.markdown("### FUNDAMENTAL ANALYSIS METRICS")
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    if 'trailingPE' in info:
                        st.metric("Trailing PE", f"{info['trailingPE']:.2f}")
                    if 'priceToBook' in info:
                        st.metric("Price to Book Ratio", f"{info['priceToBook']:.2f}")
                    if 'profitMargins' in info:
                        st.metric("Profit Margin", f"{info['profitMargins']*100:.2f}%")
                
                with metrics_col2:
                    if 'forwardPE' in info:
                        st.metric("Forward PE", f"{info['forwardPE']:.2f}")
                    if 'enterpriseToRevenue' in info:
                        st.metric("EV/R", f"{info['enterpriseToRevenue']:.2f}")
                    if 'returnOnEquity' in info:
                        st.metric("Return on Equity", f"{info['returnOnEquity']*100:.2f}%")
                
                with metrics_col3:
                    if 'enterpriseToEbitda' in info:
                        st.metric("EV/EBITDA", f"{info['enterpriseToEbitda']:.2f}")
                    if 'debtToEquity' in info:
                        st.metric("D/E", f"{info['debtToEquity']:.2f}")
                    if 'returnOnAssets' in info:
                        st.metric("ROA", f"{info['returnOnAssets']*100:.2f}%")
                
                with metrics_col4:
                    if 'dividendYield' in info and info['dividendYield'] is not None:
                        st.metric("Dividend Yield", f"{info['dividendYield']*100:.2f}%")
                    if 'payoutRatio' in info and info['payoutRatio'] is not None:
                        st.metric("Payout Ratio", f"{info['payoutRatio']*100:.2f}%")
                    if 'beta' in info:
                        st.metric("Beta", f"{info['beta']:.2f}")

                st.markdown("---")
                
                # TECHNICAL INDICATORS
                st.markdown("### TECHNICAL INDICATORS")
                
                # MOVING AVERAGES
                daily_data['MA50'] = daily_data['Close'].rolling(window=50).mean()
                daily_data['MA200'] = daily_data['Close'].rolling(window=200).mean()
                daily_data['RSI'] = calculate_rsi(daily_data['Close'])
                
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                with tech_col1:
                    st.metric("50 Day Average", f"{daily_data['MA50'][-1]:.2f}")
                with tech_col2:
                    st.metric("200 Day Average", f"{daily_data['MA200'][-1]:.2f}")
                with tech_col3:
                    st.metric("RSI (14)", f"{daily_data['RSI'][-1]:.2f}")
                with tech_col4:
                    volatility = daily_data['Close'].pct_change().std() * (252 ** 0.5) * 100
                    st.metric("Annual Volatility", f"{volatility:.2f}%")

                st.markdown("---")
                
                # Graphics
                st.markdown("### PRICE AND VOLUME CHARTS")
                
                # Dropdown for timezone selection
                time_periods = {
                    "1 Day": ("90d", "1d"), 
                    "1 Week": ("180d", "1d"),
                    "1 Month": ("730d", "1d"),
                    "1 Year": ("max", "1d")
                }
                selected_period = st.selectbox(
                    "Candle",
                    options=list(time_periods.keys()),
                    index=1,  # 1 Day is selected by default
                    key="chart_period"
                )
                
                # Import data only for charts
                period, interval = time_periods[selected_period]
                try:
                    chart_data = stock.history(period=period, interval=interval)
                    
                    if not chart_data.empty:
                        # resample data according to selected time period
                        if selected_period == "4 hour":
                            chart_data = chart_data.resample('4H').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            }).dropna()
                        elif selected_period == "1 week":
                            chart_data = chart_data.resample('W').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            }).dropna()
                        elif selected_period == "1 month":
                            chart_data = chart_data.resample('M').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            }).dropna()
                        elif selected_period == "1 year":
                            chart_data = chart_data.resample('Y').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            }).dropna()

                        # Moving Averages
                        if len(chart_data) > 50:
                            chart_data['MA50'] = chart_data['Close'].rolling(window=50).mean()
                        if len(chart_data) > 200:
                            chart_data['MA200'] = chart_data['Close'].rolling(window=200).mean()
                        
                        # Candlestick chart
                        fig = create_candlestick_chart(chart_data, stock_symbol, selected_period)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Volume chart
                        fig_volume = create_volume_chart(chart_data, stock_symbol, selected_period)
                        st.plotly_chart(fig_volume, use_container_width=True)
                    else:
                        st.warning(f"No data found for {selected_period}. Please try again.")
                except Exception as e:
                    st.warning(f" Data could not be retrieved for {selected_period}. Please select another time period.")

                # Comapany Information
                if 'longBusinessSummary' in info:
                    with st.expander("About Company"):
                        try:
                            
                            st.write(info['longBusinessSummary'])
                        except:
                            st.write("Information Not Found.")

                # Technical Analysis Summary
                with st.expander("Technical Analysis Summary"):
                    trend = "Uptrend" if daily_data['MA50'][-1] > daily_data['MA200'][-1] else "Downtrend"
                    st.write(f"{'⬆️' if daily_data['MA50'][-1] > daily_data['MA200'][-1] else '⬇️'} **Trend Status:** {trend}")
                    
                    if 'trailingPE' in info:
                        fk_status = "Low (Attractive)" if info['trailingPE'] < 10 else "High (Expensive)" if info['trailingPE'] > 20 else "Normal"
                        st.write(f"{'✅' if info['trailingPE'] < 10 else '⚠️' if info['trailingPE'] > 20 else 'ℹ️'} **F/K Status:** {fk_status}")
                    
                    rsi = daily_data['RSI'][-1]
                    rsi_status = "Oversold(Buying Opportunity)" if rsi < 30 else "Overbought(Selling Opportunity)" if rsi > 70 else "Normal"
                    st.write(f"{'✅' if 40 < rsi < 60 else '⚠️'} **RSI Status:** {rsi_status}")
                    
                    volume_change = ((daily_data['Volume'][-5:].mean() - daily_data['Volume'][-10:-5].mean()) / daily_data['Volume'][-10:-5].mean()) * 100
                    st.write(f"{'✅' if volume_change > 0 else 'ℹ️'} **Volume Change (5 Days):** {volume_change:.2f}%")

                # Raw Data
                with st.expander("Raw Data"):
                    show_dataframe(daily_data)
                    
            else:
                st.error("❌ Data not found. please enter a valid stock symbol")
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("💡 Please enter valid stock code")

        st.markdown("---")
        

    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # Update Graphic Styles
    def create_figure_layout(title):
        return dict(
            title=title,
            template="plotly",
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            width=1000,  # Fixed width
            height=600,  # Fixed height
            margin=dict(l=50, r=50, t=50, b=50)  # Margin
        )

    # Update candlestick chart creation function
    def create_candlestick_chart(hist, stock_symbol, period_text):
        fig = go.Figure()
        
        # candlesticks
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name="OHLC"
        ))
        
        # Moving Averages (if any)
        if 'MA50' in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['MA50'],
                name='50 Day Average.',
                line=dict(width=2)
            ))
        
        if 'MA200' in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['MA200'],
                name='200 Day Average.',
                line=dict(width=2)
            ))
        
        layout = create_figure_layout(f"{stock_symbol} {period_text} Price Chart")
        layout.update(
            xaxis=dict(
                rangeslider=dict(visible=False),
                type='date',
                autorange=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1D", step="day", stepmode="backward"),
                        dict(count=7, label="1W", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="all")
                    ])
                )
            ),
            yaxis=dict(
                autorange=True,
                fixedrange=False,
                title=dict(
                    text="Price",
                    standoff=10
                )
            ),
            dragmode='zoom',
            showlegend=True,
            hovermode='x unified'
        )
        fig.update_layout(layout)
        
        return fig

    # update volume chart creation function
    def create_volume_chart(hist, stock_symbol, period_text):
        fig_volume = go.Figure()
        
        # volume bars (dark blue colour)
        fig_volume.add_trace(go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name="Volume",
            marker_color='#001A6E'  # dark blue
        ))
        
        layout = create_figure_layout(f"{stock_symbol} {period_text} Trading Volume")
        layout.update(
            xaxis=dict(
                type='date',
                autorange=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1D", step="day", stepmode="backward"),
                        dict(count=7, label="1W", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="all")
                    ]),
                    
                )
            ),
            yaxis=dict(
                autorange=True,
                fixedrange=False,
                title=dict(
                    text="Transaction Volume",
                    standoff=10  # distance between y axis title and axis
                )
            ),
            dragmode='zoom',
            showlegend=True
        )
        fig_volume.update_layout(layout)
        
        return fig_volume




    def show_dataframe(df):
        st.dataframe(
            df,
            height=300, 
            use_container_width=True, 
        )


    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        stock_symbol = st.text_input("🔍Enter stock symbol (Example:TATACHEM.NS)", "").upper()

    if stock_symbol:
        analyze_stock(stock_symbol)


if selected == "Market Status":

    # Configure session with proper NSE cookies and headers
    def create_nse_session():
        session = requests.Session()
        # Initial request to set cookies
        session.get("https://www.nseindia.com/", headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }, timeout=10)
        return session

    @st.cache_data(ttl=300)
    def fetch_nse_data(symbols):
        data = []
        session = create_nse_session()
        
        for idx, symbol in enumerate(symbols):
            try:
                # Rate limiting
                if idx > 0:
                    time.sleep(0.5) # 1 second delay between requests

                # Fetch equity quote
                quote_url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
                headers = {
                    "Referer": f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}",
                    "X-Requested-With": "XMLHttpRequest",
                    "Authority": "www.nseindia.com",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
                }

                # Retry logic
                for attempt in range(3):
                    try:
                        response = session.get(quote_url, headers=headers, timeout=15)
                        response.raise_for_status()
                        quote_data = response.json()
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        time.sleep(2)
                        session = create_nse_session()

                # Fetch market status to maintain session
                session.get("https://www.nseindia.com/api/marketStatus", headers=headers, timeout=5)

                # Extract NSE data
                company_name = quote_data.get('info', {}).get('companyName', symbol)
                sector = quote_data.get('info', {}).get('industry', 'Unknown Sector')
                price_info = quote_data.get('priceInfo', {})
                p_change = price_info.get('pChange', 0)
                last_price = price_info.get('lastPrice', 0)

                # Get market cap from Yahoo Finance
                try:
                    yf_symbol = f"{symbol}.NS"
                    stock = yf.Ticker(yf_symbol)
                    market_cap = stock.info.get('marketCap', 0)
                    # Convert to Crores (1 Cr = 10^7)
                    market_cap_cr = round(max(market_cap / 1e7, 0.01), 2)
                except Exception as yf_error:
                    st.warning(f"YFinance error for {symbol}: {str(yf_error)}")
                    market_cap_cr = 0.01

                data.append({
                    "Symbol": symbol,
                    "Name": company_name,
                    "Sector": sector,
                    "MarketCap": market_cap_cr,
                    "PriceChange": round(p_change, 2),
                    "LastPrice": last_price
                })

            except Exception as e:
                st.error(f"Error fetching {symbol}: {str(e)}")
                data.append({
                    "Symbol": symbol,
                    "Name": symbol,
                    "Sector": "Error",
                    "MarketCap": 0.01,
                    "PriceChange": 0,
                    "LastPrice": 0
                })

        return pd.DataFrame(data)

    def main():
        st.title("NIFTY 50 HEATMAP")
        
        # Default equity symbols (avoid index symbols)
        default_symbols = ['TATASTEEL', 'NTPC', 'WIPRO', 'ITC', 'RELIANCE', 'SHRIRAMFIN', 'ONGC', 'COALINDIA', 'BHARTIARTL', 'INDUSINDBK', 'HINDALCO', 'KOTAKBANK', 'TATACONSUM', 'HDFCLIFE', 'LT', 'TCS', 'CIPLA', 'TRENT', 'ADANIENT', 'BAJAJFINSV', 'BRITANNIA', 'BAJFINANCE', 'TITAN', 'NESTLEIND', 'ULTRACEMCO', 'HEROMOTOCO', 'BAJAJ-AUTO', 'MARUTI', 'APOLLOHOSP', 'HDFCBANK', 'ICICIBANK', 'INFY', 'SBIN', 'HINDUNILVR', 'HCLTECH', 'SUNPHARMA', 'M&M', 'AXISBANK', 'POWERGRID', 'TATAMOTORS', 'ADANIPORTS', 'JSWSTEEL', 'ASIANPAINT', 'BEL', 'TECHM', 'GRASIM', 'EICHERMOT', 'BPCL', 'DRREDDY']

        # Symbol management
        

        # Data fetching
        with st.spinner("Fetching real-time data from NSE + Yahoo Finance..."):
            df = fetch_nse_data(default_symbols)
            valid_df = df[df['Sector'] != 'Error']
            valid_df = valid_df[valid_df['MarketCap'] > 0]

        # Filters
        with st.sidebar:
            st.header("Filters")
            sector_options = valid_df["Sector"].unique()
            selected_sectors = st.multiselect(
                "Select Sectors",
                options=sector_options,
                default=sector_options
            )
        
        filtered_df = valid_df[valid_df["Sector"].isin(selected_sectors)]

        # Display raw data
        

        # Create visualization
        if not filtered_df.empty:
            fig = px.treemap(
                filtered_df,
                path=[px.Constant("NSE Stocks"), 'Sector', 'Name'],
                values='MarketCap',
                color='PriceChange',
                color_continuous_scale='viridis',
                color_continuous_midpoint=0,
                hover_data=['MarketCap', 'PriceChange', 'LastPrice'],
                branchvalues='total'
            )

            fig.update_traces(
                texttemplate=(
                    "<b>%{label}</b><br>"
                    "M Cap: ₹%{customdata[0]:,.1f} Cr<br>"
                    "Change: %{customdata[1]:+.2f}%<br>"
                    "Price: ₹%{customdata[2]:,.1f}"
                ),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Market Cap: ₹%{customdata[0]:,.1f} Cr<br>"
                    "Price Change: %{customdata[1]:+.2f}%<br>"
                    "Last Price: ₹%{customdata[2]:,.1f}<extra></extra>"
                )
            )

            fig.update_layout(
                margin=dict(t=30, l=0, r=0, b=0),
                coloraxis_colorbar=dict(
                    title="Price Change (%)",
                    tickprefix="+",
                    ticksuffix="%"
                )
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid data available for selected filters!")

    if __name__ == "__main__":
        main()


if selected == "Portfolio Analysis and News":
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

    