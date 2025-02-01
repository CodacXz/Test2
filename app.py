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

def display_article(article, companies_df, article_idx):
    """Display a single article with analysis"""
    title = article.get('title', 'No title')
    description = article.get('description', 'No description')
    url = article.get('url', '#')
    source = article.get('source', 'Unknown')
    published_at = article.get('published_at', '')
    
    # Find mentioned companies first
    text = f"{title} {description}"
    mentioned_company = None
    
    # Find first company mentioned
    for _, row in companies_df.iterrows():
        company_name = str(row['Company_Name']).lower()
        company_code = str(row['Company_Code'])
        if company_name in text.lower() or company_code in text.lower():
            mentioned_company = {
                'name': row['Company_Name'],
                'code': company_code,
                'symbol': f"{company_code}.SR"
            }
            break
    
    # Display article content
    with st.container():
        st.markdown(f"## {title}", key=f"title_{article_idx}")
        st.write(f"Source: {source} | Published: {published_at[:16]}", key=f"meta_{article_idx}")
        st.write(description, key=f"desc_{article_idx}")
        
        # Sentiment Analysis
        sentiment, confidence = analyze_sentiment(title + " " + description)
        
        col1, col2 = st.columns(2, key=f"cols_{article_idx}")
        with col1:
            st.markdown("### Sentiment Analysis", key=f"sent_title_{article_idx}")
            st.write(f"**Sentiment:** {sentiment}", key=f"sent_value_{article_idx}")
            st.write(f"**Confidence:** {confidence:.2f}%", key=f"conf_value_{article_idx}")
        
        if mentioned_company:
            st.write("### Company Analysis", key=f"comp_title_{article_idx}")
            st.write(f"**{mentioned_company['name']} ({mentioned_company['symbol']})**", 
                    key=f"comp_name_{article_idx}")
            
            try:
                df, error = get_stock_data(mentioned_company['symbol'])
                if error:
                    st.error(error, key=f"error_{article_idx}")
                else:
                    if df is not None and not df.empty:
                        # Show metrics
                        latest_price = df['Close'][-1]
                        price_change = ((latest_price - df['Close'][-2])/df['Close'][-2]*100)
                        
                        metrics_cols = st.columns(3)
                        with metrics_cols[0]:
                            st.metric(
                                "Current Price", 
                                f"{latest_price:.2f} SAR",
                                f"{price_change:.2f}%",
                                key=f"price_{article_idx}"
                            )
                        with metrics_cols[1]:
                            st.metric(
                                "Day High", 
                                f"{df['High'][-1]:.2f} SAR",
                                key=f"high_{article_idx}"
                            )
                        with metrics_cols[2]:
                            st.metric(
                                "Day Low", 
                                f"{df['Low'][-1]:.2f} SAR",
                                key=f"low_{article_idx}"
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
                            title=None,
                            yaxis_title='Price (SAR)',
                            xaxis_title='Date',
                            template='plotly_dark',
                            height=400,
                            margin=dict(t=0)
                        )
                        
                        st.plotly_chart(
                            fig, 
                            key=f"chart_{article_idx}",
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Error analyzing {mentioned_company['name']}: {str(e)}", 
                        key=f"analysis_error_{article_idx}")
        
        st.markdown(f"[Read full article]({url})", key=f"url_{article_idx}")
        st.markdown("---", key=f"divider_{article_idx}")

def main():
    st.title("Saudi Stock Market News", key="main_title")
    st.write("Real-time news analysis for Saudi stock market", key="main_desc")
    
    # Sidebar settings
    st.sidebar.title("Settings", key="settings_title")
    uploaded_file = st.sidebar.file_uploader(
        "Upload companies file (optional)", 
        type=['csv'],
        key="file_uploader"
    )
    
    # Load company data
    companies_df = load_company_data(uploaded_file)
    if companies_df.empty:
        st.warning("⚠️ No company data loaded", key="data_warning")
    else:
        st.sidebar.success(f"✅ Loaded {len(companies_df)} companies", key="data_success")
    
    # Date selection
    default_date = datetime.now() - timedelta(days=7)
    published_after = st.date_input(
        "Show news published after:", 
        value=default_date,
        key="date_input"
    )
    published_after_iso = published_after.isoformat() + "T00:00:00"
    
    # Number of articles
    limit = st.sidebar.slider(
        "Number of articles", 
        1, 3, 3,
        key="article_limit"
    )
    
    # Fetch news
    if st.button("Fetch News", key="fetch_button", use_container_width=True):
        with st.spinner("Fetching latest news..."):
            articles = fetch_news(published_after_iso, limit)
            
            if articles:
                st.success(f"Found {len(articles)} articles", key="fetch_success")
                for idx, article in enumerate(articles):
                    display_article(article, companies_df, idx)
            else:
                st.warning("No articles found for the selected period", key="fetch_warning")

if __name__ == "__main__":
    main()
