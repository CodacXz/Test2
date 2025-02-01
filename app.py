import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import plotly.graph_objects as go
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# API Configuration
NEWS_API_URL = "https://api.marketaux.com/v1/news/all"
API_TOKEN = st.secrets["general"]["MARKETAUX_API_KEY"]

@st.cache_data
def load_company_data(uploaded_file=None):
    """Load company data from uploaded file or default GitHub URL"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            github_url = "https://raw.githubusercontent.com/CodacXz/Test/main/saudi_companies.csv?raw=true"
            df = pd.read_csv(github_url)
        
        # Clean and prepare data
        df['Company_Name'] = df['Company_Name'].str.strip()
        df['Company_Code'] = df['Company_Code'].astype(str).str.zfill(4)
        
        return df
    except Exception as e:
        st.error(f"Error loading company data: {str(e)}")
        return pd.DataFrame()

def find_companies_in_text(text, companies_df):
    """Find unique companies mentioned in the text"""
    if not text or companies_df.empty:
        return []
    
    text = text.lower()
    seen_companies = set()  # Track unique companies
    mentioned_companies = []
    
    for _, row in companies_df.iterrows():
        company_name = str(row['Company_Name']).lower()
        company_code = str(row['Company_Code'])
        
        # Only add each company once
        if (company_name in text or company_code in text) and company_code not in seen_companies:
            seen_companies.add(company_code)
            mentioned_companies.append({
                'name': row['Company_Name'],
                'code': company_code,
                'symbol': f"{company_code}.SR"
            })
    
    return mentioned_companies

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    analyzer = SentimentIntensityAnalyzer()
    try:
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = "POSITIVE"
            confidence = min((compound + 1) / 2, 1.0)
        elif compound <= -0.05:
            sentiment = "NEGATIVE"
            confidence = min(abs(compound), 1.0)
        else:
            sentiment = "NEUTRAL"
            confidence = 0.5
        
        return sentiment, confidence * 100
    except Exception as e:
        st.warning(f"Error in sentiment analysis: {str(e)}")
        return "NEUTRAL", 50.0

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
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data:
            st.error(f"API Error: {data['error']['message']}")
            return []
            
        return data.get("data", [])
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

def get_stock_data(symbol, period='1mo'):
    """Fetch stock data and calculate technical indicators"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        
        if df.empty:
            return None, f"No stock data available for {symbol}"
        
        # Calculate indicators
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        rsi = RSIIndicator(df['Close'])
        df['RSI'] = rsi.rsi()
        
        bb = BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        
        return df, None
    except Exception as e:
        return None, f"Error fetching data for {symbol}: {str(e)}"

def display_article(article, companies_df):
    """Display a single article with analysis"""
    title = article.get('title', 'No title')
    description = article.get('description', 'No description')
    url = article.get('url', '#')
    source = article.get('source', 'Unknown')
    published_at = article.get('published_at', '')
    
    # Find mentioned companies first
    text = f"{title} {description}"
    mentioned_companies = []
    seen = set()
    
    # First pass: collect all unique companies
    for _, row in companies_df.iterrows():
        company_name = str(row['Company_Name']).lower()
        company_code = str(row['Company_Code'])
        if (company_name in text.lower() or company_code in text.lower()) and company_code not in seen:
            seen.add(company_code)
            mentioned_companies.append({
                'name': row['Company_Name'],
                'code': company_code,
                'symbol': f"{company_code}.SR"
            })
    
    # Skip articles with more than 2 companies
    if len(mentioned_companies) > 2:
        st.info(f"Skipping article '{title}' - too many companies mentioned ({len(mentioned_companies)})")
        return
    
    # Display article content
    st.markdown(f"## {title}")
    st.write(f"Source: {source} | Published: {published_at[:16]}")
    st.write(description)
    
    # Sentiment Analysis
    sentiment, confidence = analyze_sentiment(title + " " + description)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Sentiment Analysis")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}%")
    
    if mentioned_companies:
        # Show companies mentioned
        st.write("### Companies Mentioned")
        for company in mentioned_companies:
            st.write(f"- {company['name']} ({company['symbol']})")
        
        # Show stock analysis one by one
        st.write("### Stock Analysis")
        for company in mentioned_companies:
            try:
                df, error = get_stock_data(company['symbol'])
                if error:
                    st.error(error)
                    continue
                
                if df is not None and not df.empty:
                    # Create a unique container for each company
                    container_key = f"container_{company['code']}_{hash(title)}"
                    with st.container():
                        st.subheader(f"{company['name']} Stock Price")
                        
                        # Show current price and change first
                        latest_price = df['Close'][-1]
                        price_change = ((latest_price - df['Close'][-2])/df['Close'][-2]*100)
                        st.metric(
                            "Current Price", 
                            f"{latest_price:.2f} SAR",
                            f"{price_change:.2f}%"
                        )
                        
                        # Create chart
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name='Price'
                        ))
                        
                        fig.update_layout(
                            title=None,  # Remove title to avoid duplication
                            yaxis_title='Price (SAR)',
                            xaxis_title='Date',
                            template='plotly_dark',
                            height=400,
                            margin=dict(t=0)  # Remove top margin
                        )
                        
                        st.plotly_chart(fig, key=container_key, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error analyzing {company['name']}: {str(e)}")
    
    st.markdown(f"[Read full article]({url})")
    st.markdown("---")

def main():
    st.title("Saudi Stock Market News")
    st.write("Real-time news analysis for Saudi stock market")
    
    # Sidebar settings
    st.sidebar.title("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload companies file (optional)", type=['csv'])
    
    # Load company data
    companies_df = load_company_data(uploaded_file)
    if companies_df.empty:
        st.warning("⚠️ No company data loaded")
    else:
        st.sidebar.success(f"✅ Loaded {len(companies_df)} companies")
    
    # Date selection
    default_date = datetime.now() - timedelta(days=7)
    published_after = st.date_input("Show news published after:", value=default_date)
    published_after_iso = published_after.isoformat() + "T00:00:00"
    
    # Number of articles
    limit = st.sidebar.slider("Number of articles", 1, 3, 3)
    
    # Fetch news
    if st.button("Fetch News"):
        with st.spinner("Fetching latest news..."):
            articles = fetch_news(published_after_iso, limit)
            
            if articles:
                st.success(f"Found {len(articles)} articles")
                for article in articles:
                    display_article(article, companies_df)
            else:
                st.warning("No articles found for the selected period")

if __name__ == "__main__":
    main()
