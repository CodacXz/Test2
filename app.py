import requests
import streamlit as st
import pandas as pd
import io
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import plotly.graph_objects as go
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# API Configuration
NEWS_API_URL = "https://api.marketaux.com/v1/news/all"
API_TOKEN = "W737FzQuSSOm3MyYYLJ1kt7csT8NOwxl2WL7Gl6x"  # Temporary direct key for testing

# GitHub raw file URL for the companies CSV file
GITHUB_CSV_URL = "https://raw.githubusercontent.com/CodacXz/Test/main/saudi_companies.csv?raw=true"

@st.cache_data
def load_company_data(uploaded_file=None):
    """Load and cache company data from either uploaded file or GitHub"""
    try:
        if uploaded_file is not None:
            # Load from uploaded file
            df = pd.read_csv(uploaded_file, encoding='utf-8', sep=',', engine='python')
        else:
            # Load from GitHub
            response = requests.get(GITHUB_CSV_URL, timeout=10)
            response.raise_for_status()
            # Use pandas read_csv with explicit parameters
            df = pd.read_csv(
                GITHUB_CSV_URL,
                encoding='utf-8',
                sep=',',
                engine='python',
                on_bad_lines='skip'  # Skip problematic lines
            )
        
        # Convert company names and codes to lowercase for better matching
        df['Company_Name_Lower'] = df['Company_Name'].str.lower()
        # Convert company codes to strings and ensure they're padded to 4 digits
        df['Company_Code'] = df['Company_Code'].astype(str).str.zfill(4)
        
        # Log the number of companies loaded
        st.sidebar.success(f"✅ Successfully loaded {len(df)} companies")
        return df
    except Exception as e:
        st.error(f"Error loading company data: {e}")
        # Print more detailed error information
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

def find_companies_in_text(text, companies_df):
    """Find all companies mentioned in the text"""
    if companies_df.empty:
        return []
    
    text = text.lower()
    mentioned_companies = []
    
    # Search for each company in the text
    for _, row in companies_df.iterrows():
        company_name = str(row['Company_Name']).lower()
        company_code = str(row['Company_Code']).zfill(4)  # Ensure 4-digit format
        
        # Check if company name or code is in text
        if company_name in text or company_code in text:
            mentioned_companies.append({
                'code': company_code,  # This will be 4 digits
                'name': row['Company_Name'],
                'symbol': f"{company_code}.SR"  # Add Saudi market suffix
            })
    
    return mentioned_companies

def find_company_code(text, companies_df):
    """Find company code from news text"""
    if companies_df.empty:
        return None, None
    
    text_lower = text.lower()
    
    # Try to find any company name in the text
    for _, row in companies_df.iterrows():
        if row['Company_Name_Lower'] in text_lower:
            return row['Company_Code'], row['Company_Name']
    
    return None, None

def analyze_sentiment(text):
    """Analyze sentiment of text using VADER and financial keywords"""
    # Initialize VADER
    analyzer = SentimentIntensityAnalyzer()
    
    # Add financial-specific words to VADER lexicon
    analyzer.lexicon.update({
        'fine': -3.0,          # Very negative
        'penalty': -3.0,       # Very negative
        'violation': -3.0,     # Very negative
        'regulatory': -2.0,    # Negative
        'investigation': -2.0, # Negative
        'lawsuit': -2.0,       # Negative
        'corrective': -1.0,    # Slightly negative
        'inaccurate': -1.0,    # Slightly negative
        'misleading': -2.0     # Negative
    })
    
    try:
        # Get sentiment scores
        scores = analyzer.polarity_scores(text)
        compound_score = scores['compound']  # This is the normalized combined score
        
        # Convert to sentiment and confidence
        if compound_score >= 0.05:
            return "POSITIVE", min((compound_score + 1) / 2, 1.0)
        elif compound_score <= -0.05:
            return "NEGATIVE", min(abs(compound_score), 1.0)
        else:
            return "NEUTRAL", 0.5
        
    except Exception as e:
        st.warning(f"Sentiment analysis failed: {e}")
        return "NEUTRAL", 0.5

def fetch_news(published_after, limit=3):
    """Fetch news articles from MarketAux API"""
    params = {
        "api_token": API_TOKEN,
        "countries": "sa",
        "filter_entities": "true",
        "limit": limit,
        "published_after": published_after
    }
    
    try:
        st.write("Fetching news with params:", params)  # Debug
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        st.write("API Response:", data)  # Debug
        if 'error' in data:
            st.error(f"API Error: {data['error']['message']}")
            return []
        articles = data.get("data", [])
        st.write(f"Number of articles received: {len(articles)}")  # Debug
        return articles
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def get_stock_data(symbol, period='1mo'):
    """Fetch stock data and calculate technical indicators"""
    try:
        # Format symbol for Saudi market
        symbol = str(symbol).zfill(4) + ".SR"  # Saudi market uses 4 digits + .SR
        
        # Get stock data
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        if df.empty:
            return None, f"No stock data available for {symbol}"
        
        # Calculate technical indicators
        # MACD
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # RSI
        rsi = RSIIndicator(df['Close'])
        df['RSI'] = rsi.rsi()
        
        # Bollinger Bands
        bb = BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        
        return df, None
    except Exception as e:
        return None, f"Error fetching data for {symbol}: {str(e)}"

def analyze_technical_indicators(df):
    """Analyze technical indicators and generate trading signals"""
    latest = df.iloc[-1]
    signals = []
    
    # MACD Analysis
    if latest['MACD'] > latest['MACD_Signal']:
        signals.append(("MACD", "BULLISH", "MACD line above signal line"))
    else:
        signals.append(("MACD", "BEARISH", "MACD line below signal line"))
    
    # RSI Analysis
    if latest['RSI'] > 70:
        signals.append(("RSI", "BEARISH", "Overbought condition (RSI > 70)"))
    elif latest['RSI'] < 30:
        signals.append(("RSI", "BULLISH", "Oversold condition (RSI < 30)"))
    else:
        signals.append(("RSI", "NEUTRAL", f"Normal range (RSI: {latest['RSI']:.2f})"))
    
    # Bollinger Bands Analysis
    close = latest['Close']
    if close > latest['BB_upper']:
        signals.append(("Bollinger Bands", "BEARISH", "Price above upper band"))
    elif close < latest['BB_lower']:
        signals.append(("Bollinger Bands", "BULLISH", "Price below lower band"))
    else:
        signals.append(("Bollinger Bands", "NEUTRAL", "Price within bands"))
    
    return signals

def plot_stock_analysis(df, company_name, symbol):
    """Create an interactive plot with price and indicators"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper',
                            line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower',
                            line=dict(color='gray', dash='dash')))
    
    fig.update_layout(
        title=f'{company_name} ({symbol}) - Price and Technical Indicators',
        yaxis_title='Price (SAR)',
        xaxis_title='Date',
        template='plotly_dark'
    )
    
    # Use a unique key for each chart based on company symbol and timestamp
    chart_key = f"stock_chart_{symbol}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}"
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

def display_article(article, companies_df):
    """Display news article with sentiment and technical analysis"""
    title = article.get("title", "No title")
    description = article.get("description", "No description")
    url = article.get("url", "#")
    source = article.get("source", "Unknown")
    published_at = article.get("published_at", "")
    
    # Generate a unique ID for this article
    article_id = pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')
    
    st.markdown(f"## {title}")
    
    # Display source and date
    st.write(f"Source: {source} | Published: {published_at[:16]}")
    
    # Display description
    st.write(description[:200] + "..." if len(description) > 200 else description)
    
    # Sentiment Analysis
    sentiment, confidence = analyze_sentiment(title + " " + description)
    
    # Create columns for analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sentiment Analysis")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}%")
    
    # Find mentioned companies and deduplicate based on symbol
    mentioned_companies = find_companies_in_text(title + " " + description, companies_df)
    if mentioned_companies:
        # Create a dictionary to deduplicate companies by symbol
        unique_companies = {}
        for company in mentioned_companies:
            if company['symbol'] not in unique_companies:
                unique_companies[company['symbol']] = company
        
        # Display companies mentioned
        st.markdown("### Companies Mentioned")
        for company in mentioned_companies:
            st.write(f"{company['name']} ({company['symbol']})")
        
        # Analyze each unique company
        st.markdown("### Stock Analysis")
        for idx, company in enumerate(unique_companies.values()):
            # Create a unique key for the expander based on article and company
            expander_key = f"expander_{article_id}_{company['symbol']}_{idx}"
            with st.expander(f"{company['name']} ({company['symbol']})", key=expander_key):
                # Get stock data and technical analysis
                df, error = get_stock_data(company['code'])
                if error:
                    st.error(f"Error fetching stock data: {error}")
                    continue
                
                if df is not None:
                    # Show current stock price
                    latest_price = df['Close'][-1]
                    price_key = f"price_{article_id}_{company['symbol']}_{idx}"
                    st.metric(
                        "Current Price",
                        f"{latest_price:.2f} SAR",
                        f"{((latest_price - df['Close'][-2])/df['Close'][-2]*100):.2f}%",
                        key=price_key
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        high_key = f"high_{article_id}_{company['symbol']}_{idx}"
                        st.metric("Day High", f"{df['High'][-1]:.2f} SAR", key=high_key)
                    with col2:
                        low_key = f"low_{article_id}_{company['symbol']}_{idx}"
                        st.metric("Day Low", f"{df['Low'][-1]:.2f} SAR", key=low_key)
                    
                    # Create a unique key for the chart
                    chart_key = f"chart_{article_id}_{company['symbol']}_{idx}"
                    
                    # Plot stock chart with the unique key
                    fig = go.Figure()
                    
                    # Add candlestick chart
                    fig.add_trace(
                        go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name='Price'
                        )
                    )
                    
                    # Add volume bars
                    fig.add_trace(
                        go.Bar(
                            x=df.index,
                            y=df['Volume'],
                            name='Volume',
                            yaxis='y2',
                            opacity=0.3
                        )
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f'{company["name"]} ({company["symbol"]}) - Price and Volume',
                        yaxis_title='Price (SAR)',
                        yaxis2=dict(
                            title='Volume',
                            overlaying='y',
                            side='right'
                        ),
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)
                    
                    # Technical Analysis Signals
                    signals_key = f"signals_{article_id}_{company['symbol']}_{idx}"
                    st.markdown("### Technical Analysis Signals", key=signals_key)
                    signals = analyze_technical_indicators(df)
                    
                    # Create a clean table for signals
                    signal_df = pd.DataFrame(signals, columns=['Indicator', 'Signal', 'Reason'])
                    table_key = f"table_{article_id}_{company['symbol']}_{idx}"
                    st.table(signal_df, key=table_key)
                    
                    # Combined Analysis
                    analysis_key = f"analysis_{article_id}_{company['symbol']}_{idx}"
                    st.markdown("### Combined Analysis", key=analysis_key)
                    tech_sentiment = sum(1 if signal[1] == "BULLISH" else -1 if signal[1] == "BEARISH" else 0 for signal in signals)
                    news_sentiment_score = 1 if sentiment == "POSITIVE" else -1 if sentiment == "NEGATIVE" else 0
                    
                    combined_score = (tech_sentiment + news_sentiment_score) / (len(signals) + 1)
                    
                    score_key = f"score_{article_id}_{company['symbol']}_{idx}"
                    if combined_score > 0.3:
                        st.success("🟢 Overall Bullish: Technical indicators and news sentiment suggest positive momentum", key=score_key)
                    elif combined_score < -0.3:
                        st.error("🔴 Overall Bearish: Technical indicators and news sentiment suggest negative pressure", key=score_key)
                    else:
                        st.warning("🟡 Neutral: Mixed signals from technical indicators and news sentiment", key=score_key)
                    
                    # Volume Analysis
                    volume_key = f"volume_{article_id}_{company['symbol']}_{idx}"
                    avg_volume = df['Volume'].mean()
                    latest_volume = df['Volume'][-1]
                    volume_change = ((latest_volume - avg_volume) / avg_volume) * 100
                    
                    st.markdown("### Volume Analysis", key=volume_key)
                    volume_metric_key = f"volume_metric_{article_id}_{company['symbol']}_{idx}"
                    st.metric(
                        "Trading Volume",
                        f"{int(latest_volume):,}",
                        f"{volume_change:.1f}% vs 30-day average",
                        key=volume_metric_key
                    )
    
    # Article link
    link_key = f"link_{article_id}"
    st.markdown(f"[Read full article]({url})", key=link_key)
    st.markdown("---")

def main():
    st.title("Saudi Stock Market News")
    st.write("Real-time news analysis for Saudi stock market")

    # File upload option in sidebar
    st.sidebar.title("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload companies file (optional)", type=['csv'])
    
    # Load company data
    companies_df = load_company_data(uploaded_file)
    if companies_df.empty:
        st.warning("⚠️ No company data loaded. Either upload a CSV file or update the GitHub URL in the code.")
    else:
        st.sidebar.success(f"✅ Loaded {len(companies_df)} companies")

    # Rest of the settings
    limit = st.sidebar.slider("Number of articles", 1, 3, 3)
    
    # Date selection
    default_date = datetime.now() - timedelta(days=7)
    published_after = st.date_input("Show news published after:", value=default_date)
    published_after_iso = published_after.isoformat() + "T00:00:00"

    # Fetch and display news
    if st.button("Fetch News"):
        with st.spinner("Fetching latest news..."):
            news_articles = fetch_news(published_after_iso, limit)
            
            if news_articles:
                st.success(f"Found {len(news_articles)} articles")
                for article in news_articles:
                    display_article(article, companies_df)
            else:
                st.warning("No news articles found for the selected date range")

    # App information
    st.sidebar.markdown("---")
    st.sidebar.write("App Version: 1.0.5")
    
    # API status
    st.sidebar.success("✅ API Token loaded")
    
    # Add GitHub information
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### How to use company data:
    1. **Option 1:** Upload CSV file using the uploader above
    2. **Option 2:** Add file to GitHub and update `GITHUB_CSV_URL`
    
    CSV file format:
    ```
    Company_Code,Company_Name
    1010,Riyad Bank
    1020,Bank Aljazira
    ...
    ```
    """)

if __name__ == "__main__":
    main()
