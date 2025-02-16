import streamlit as st
import yfinance as yf
import requests
from requests.exceptions import RequestException
import time
import pandas as pd
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
    options=["Home","Market Status","Portfolio Analysis"],
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


if selected == "Portfolio Analysis":
    # Portfolio Analysis Page Content
    
    # Configure Google Gemini API
    genai.configure(api_key="AIzaSyDrPnsBMj74vzUKb-jTG0jJSIi8Xk7pqWE")  # Set your Google Gemini API key here
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Set the style for matplotlib
    plt.style.use('dark_background')

    # Function to fetch and process data
    def fetch_data(tickers, start_date, end_date):
        Returns = []
        for ticker in tickers:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            df['DailyReturn'] = np.log(df['Adj Close']).diff()
            Returns.append(df)
        return Returns

    # Function to calculate portfolio metrics
    def calculate_metrics(Returns, weights):
        annual_returns = []
        for i, df in enumerate(Returns):
            initial = df['Adj Close'].iloc[0]
            last = df['Adj Close'].iloc[-1]
            annual_returns.append((last / initial - 1))
        
        portfolio_return = sum(np.array(annual_returns) * np.array(weights))
        portfolio_std_dev = np.sqrt(np.dot(weights, np.dot(np.cov([df['DailyReturn'].dropna() for df in Returns]), weights)))
        
        return annual_returns, portfolio_return, portfolio_std_dev

    # Definitions of financial metrics
    # Beta: A measure of a stock's volatility in relation to the market.
    # Standard Deviation: A measure of the amount of variation or dispersion of a set of values.
    # Expected Portfolio Return (CAPM): The return expected from an asset based on its risk compared to the market.
    # Sharpe Ratio: A measure of risk-adjusted return.
    # Jensen's Alpha: A measure of the excess return of a portfolio over the expected return.
    # Treynor Ratio: A measure of returns earned in excess of that which could have been earned on a riskless investment per each unit of market risk.

    # Function to calculate CAPM
    def CAPM(beta, market_return):
        return Rf + beta * (market_return - Rf)

    # Function to calculate Sharpe Ratio
    def Sharpe(portfolio_return, portfolio_std_dev, risk_free_rate):
        return (portfolio_return - risk_free_rate) / portfolio_std_dev

    # Function to calculate Jensen's Alpha
    def JensenAlpha(portfolio_return, beta, market_return):
        return portfolio_return - CAPM(beta, market_return)

    # Function to calculate Treynor Ratio
    def Treynor(portfolio_return, beta, risk_free_rate):
        return (portfolio_return - risk_free_rate) / beta

    # Function to calculate regression for beta
    def calculate_beta(Returns, market_returns):
        pfdr = pd.DataFrame({'WeightedDailyReturn': [0] * len(Returns[0])})
        pfdr.index = Returns[0].index

        for i in range(len(Returns)):
            pfdr['WeightedDailyReturn'] += Returns[i]['DailyReturn'].values * weights[i]

        model = sm.OLS(pfdr['WeightedDailyReturn'].dropna(), market_returns.dropna()).fit()
        return model.params.values[0], model.summary()

    # Function to generate AI-based portfolio summary using Google Gemini
    def generate_ai_summary(portfolio_return, portfolio_std_dev, beta, expected_return, sharpe_ratio, jensen_alpha, tickers, weights):
        prompt = f"Generate a detailed summary of a portfolio with the following metrics:\n" \
                f"Portfolio Return: {round(portfolio_return * 100, 2)}%\n" \
                f"Portfolio Risk (Standard Deviation): {round(portfolio_std_dev * 100, 2)}%\n" \
                f"Portfolio Beta: {round(beta, 3)}\n" \
                f"Expected Return (CAPM): {round(expected_return * 100, 2)}%\n" \
                f"Sharpe Ratio: {round(sharpe_ratio, 3)}\n" \
                f"Jensen's Alpha: {round(jensen_alpha * 100, 2)}%\n" \
                f"Asset Allocation: {', '.join([f'{tickers[i]}: {weights[i] * 100:.2f}%' for i in range(len(tickers))])}\n" \
                f"Provide insights on performance, risk profile, and overall assessment."
        
        response = model.generate_content(prompt)  # Call the Gemini API to get the summary
        return response.text  # Return the generated summary

    # Streamlit app layout
    st.title("Python Portfolio Performance Analyzer")
    # Adding the logo

    # User inputs
    tickers = st.text_input("Enter stock tickers (comma-separated)", "RELIANCE.NS, HCLTECH.NS, BHARTIARTL.NS, SBIN.NS, SUNPHARMA.NS")
    weights_input = st.text_input("Enter weights (comma-separated)", "0.15, 0.15, 0.25, 0.15, 0.30")
    Rf = 0.06  # Risk-free rate

    # Process user inputs
    tickers = [ticker.strip() for ticker in tickers.split(',')]
    weights = [float(weight) for weight in weights_input.split(',')]

    # Dates for data retrieval
    end_date = date.today()
    start_date = end_date - timedelta(days=366)

    # Fetch data
    Returns = fetch_data(tickers, start_date, end_date)

    # Calculate metrics
    annual_returns, portfolio_return, portfolio_std_dev = calculate_metrics(Returns, weights)

    # Fetch market data (Nifty)
    market_data = fetch_data(['^NSEI'], start_date, end_date)
    market_returns = market_data[0]['DailyReturn']

    # Calculate beta
    beta, beta_summary = calculate_beta(Returns, market_returns)

    # Calculate performance metrics
    expected_return = CAPM(beta, market_returns.mean())
    sharpe_ratio = Sharpe(portfolio_return, portfolio_std_dev, Rf)
    jensen_alpha = JensenAlpha(portfolio_return, beta, market_returns.mean())
    treynor_ratio = Treynor(portfolio_return, beta, Rf)

    # Display results
    st.write(f"The Portfolio Return is {round(portfolio_return * 100, 2)}%")
    st.write(f"The Portfolio Risk (Standard Deviation) is {round(portfolio_std_dev * 100, 2)}%")
    st.write(f"The Portfolio Beta is {round(beta, 3)}")
    st.write(f"The Expected Portfolio Return (CAPM) is {round(expected_return * 100, 2)}%")
    st.write(f"The Sharpe Ratio is {round(sharpe_ratio, 3)}")
    st.write(f"The Jensen's Alpha is {round(jensen_alpha * 100, 2)}%")
    st.write(f"The Treynor Ratio is {round(treynor_ratio, 3)}")

    # Visualization
    st.subheader("Portfolio Composition")
    plt.figure(figsize=(10, 6))
    plt.pie(weights, labels=tickers, autopct='%1.1f%%')
    plt.title('Portfolio Composition')
    st.pyplot(plt)

    # Display regression summary
    st.subheader("Regression Summary for Beta")
    st.write(beta_summary)

    # Additional visualizations for daily returns vs Nifty
    for i in range(len(Returns)):
        plt.figure(figsize=(10, 6))
        plt.plot(Returns[i]['DailyReturn'], label=tickers[i])
        plt.plot(market_returns, label='Nifty', color='orange')
        plt.title(f'Daily Returns of {tickers[i]} vs Nifty')
        plt.xlabel('Date')
        plt.ylabel('Daily Returns')
        plt.legend()
        st.pyplot(plt)

    # Calculate return contribution
    return_contribution = [weights[i] * annual_returns[i] for i in range(len(annual_returns))]

    # Visualization for Return Contribution using Bar Chart
    st.subheader("Return Contribution to the Portfolio")
    plt.figure(figsize=(10, 6))
    plt.bar(tickers, return_contribution, color='skyblue')
    plt.title('Return Contribution to the Portfolio')
    plt.xlabel('Stocks')
    plt.ylabel('Return Contribution')
    st.pyplot(plt)

    # Generate and display AI-based portfolio summary
    st.subheader("Portfolio Analysis Summary (AI based)")
    summary = generate_ai_summary(portfolio_return, portfolio_std_dev, beta, expected_return, sharpe_ratio, jensen_alpha, tickers, weights)
    st.write(summary)

    # Additional visualizations can be added here...
