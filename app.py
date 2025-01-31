import requests
import streamlit as st
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import yfinance as yf
import pandas as pd
import ta  # Technical Analysis library
import plotly.graph_objects as go
import json

# API Configuration
NEWS_API_URL = "https://api.marketaux.com/v1/news/all"
API_TOKEN = st.secrets["STOCKDATA_API_TOKEN"]

# Add this after the API configuration
MOCK_NEWS_DATA = [
    {
        "title": "CMA Imposes Fine on Arabia Insurance Cooperative Co.",
        "description": "The Capital Market Authority (CMA) board decided to impose a SAR 10,000 fine on Arabia Insurance Cooperative Co. for violating Paragraph (A) of Article 78 of the Rules on the Offer of Securities and Continuing Obligations.",
        "source": "Mock Data",
        "url": "#",
        "published_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    },
    {
        "title": "Saudi Stock Market Shows Strong Performance",
        "description": "The Saudi stock market (Tadawul) showed positive performance with increased trading volume and market capitalization.",
        "source": "Mock Data",
        "url": "#",
        "published_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    }
]

@st.cache_resource
def load_finbert():
    """Load FinBERT model for financial sentiment analysis"""
    model_name = "ProsusAI/finbert"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

@st.cache_resource
def load_roberta():
    """Load RoBERTa model for general sentiment analysis"""
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def analyze_sentiment(text):
    """Enhanced sentiment analysis using multiple models"""
    try:
        # Initialize all analyzers
        vader_analyzer = SentimentIntensityAnalyzer()
        finbert = load_finbert()
        roberta = load_roberta()
        
        # Custom financial terms with valence scores
        financial_terms = {
            'fine': -2.0,
            'penalty': -2.0,
            'violation': -2.0,
            'violating': -2.0,
            'failed': -1.5,
            'failure': -1.5,
            'loss': -1.0,
            'decline': -1.0,
            'decrease': -1.0,
            'increase': 1.0,
            'growth': 1.5,
            'profit': 1.5,
            'success': 1.5,
            'successful': 1.5,
            'positive': 1.0,
            'negative': -1.0,
            'correction': -1.0,
            'corrective': -1.0,
            'impose': -1.5,
            'imposed': -1.5,
            'dividend': 1.0,
            'earnings': 0.5,
            'revenue': 0.5,
            'bankruptcy': -2.0,
            'debt': -0.5,
            'lawsuit': -1.5,
            'investigation': -1.0,
            'merger': 0.5,
            'acquisition': 0.5,
            'partnership': 1.0,
            'regulatory': -0.5,
            'compliance': 0.5,
            'non-compliance': -1.5,
            'outperform': 1.5,
            'underperform': -1.5,
            'sanction': -1.5,
            'fraud': -2.0,
            'scandal': -2.0,
            'default': -2.0,
            'upgrade': 1.5,
            'downgrade': -1.5
        }
        
        # Update VADER lexicon
        vader_analyzer.lexicon.update(financial_terms)
        
        # Get VADER scores
        vader_scores = vader_analyzer.polarity_scores(text)
        
        # Get FinBERT prediction
        finbert_result = finbert(text)[0]
        
        # Get RoBERTa prediction
        roberta_result = roberta(text)[0]
        
        # Combine scores with weights
        vader_weight = 0.3
        finbert_weight = 0.5  # Higher weight for financial-specific model
        roberta_weight = 0.2
        
        # Normalize scores
        vader_compound = vader_scores['compound']
        finbert_score = map_finbert_score(finbert_result)
        roberta_score = map_roberta_score(roberta_result)
        
        # Calculate weighted compound score
        compound_score = (vader_compound * vader_weight +
                        finbert_score * finbert_weight +
                        roberta_score * roberta_weight)
        
        # Additional financial context analysis
        text_lower = text.lower()
        strong_negative_terms = ['fine', 'penalty', 'violation', 'bankruptcy', 'lawsuit', 'fraud', 'scandal']
        has_strong_negative = any(term in text_lower for term in strong_negative_terms)
        
        if has_strong_negative:
            compound_score = min(compound_score - 0.3, -0.2)
        
        # Determine final sentiment and confidence
        if compound_score >= 0.05:
            sentiment = "POSITIVE"
            confidence = (compound_score + 1) / 2
        elif compound_score <= -0.05:
            sentiment = "NEGATIVE"
            confidence = abs(compound_score)
        else:
            sentiment = "NEUTRAL"
            confidence = 0.5
            
        # Detailed metrics
        details = {
            'vader': vader_scores,
            'finbert': finbert_result,
            'roberta': roberta_result,
            'compound': compound_score,
            'pos': vader_scores['pos'],
            'neu': vader_scores['neu'],
            'neg': vader_scores['neg']
        }
            
        return sentiment, confidence, details
        
    except Exception as e:
        st.warning(f"Sentiment analysis failed: {e}")
        return "NEUTRAL", 0.5, {'pos': 0, 'neu': 1, 'neg': 0, 'compound': 0}

def map_finbert_score(result):
    """Map FinBERT result to [-1, 1] range"""
    label_map = {
        'positive': 1,
        'negative': -1,
        'neutral': 0
    }
    return label_map[result['label']] * result['score']

def map_roberta_score(result):
    """Map RoBERTa result to [-1, 1] range"""
    label_map = {
        'LABEL_2': 1,    # Positive
        'LABEL_1': 0,    # Neutral
        'LABEL_0': -1    # Negative
    }
    return label_map[result['label']] * result['score']

def fetch_news(published_after, limit=10):
    """Fetch news articles from the API with enhanced error handling"""
    # Ensure limit doesn't exceed plan restrictions
    api_limit = 3  # Current plan limit
    actual_limit = min(limit, api_limit)
    
    params = {
        "countries": "sa",
        "filter_entities": "true",
        "limit": actual_limit,  # Use adjusted limit
        "published_after": published_after,
        "api_token": API_TOKEN
    }
    
    try:
        # Add timeout and retry mechanism
        for attempt in range(3):  # Try up to 3 times
            try:
                response = requests.get(NEWS_API_URL, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                # Handle API warnings
                if "warnings" in data:
                    for warning in data["warnings"]:
                        st.warning(f"API Warning: {warning}")
                
                # Show meta information
                if "meta" in data:
                    st.info(f"Found {data['meta']['found']} articles, showing {data['meta']['returned']}")
                
                return data.get("data", [])
                
            except requests.exceptions.Timeout:
                if attempt == 2:  # Last attempt
                    st.error("API request timed out. Please try again later.")
                continue
            except requests.exceptions.HTTPError as http_err:
                if response.status_code == 521:
                    st.error("The news API server is currently down. Please try again later.")
                elif response.status_code == 429:
                    st.error("API rate limit exceeded. Please wait a moment and try again.")
                else:
                    st.error(f"HTTP Error: {http_err}")
                break
            except requests.exceptions.RequestException:
                continue
            
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        st.info("Try refreshing the page or waiting a few minutes before trying again.")
    
    return []

def analyze_article_sentiment(article):
    """Analyze sentiment of both title and description with weighted combination"""
    title = article.get("title", "")
    description = article.get("description", "")
    
    # Analyze both title and description
    title_sentiment, title_confidence, title_details = analyze_sentiment(title) if title else ("NEUTRAL", 0.5, None)
    desc_sentiment, desc_confidence, desc_details = analyze_sentiment(description) if description else ("NEUTRAL", 0.5, None)
    
    # Weight the scores (title: 30%, description: 70%)
    title_weight = 0.3
    desc_weight = 0.7
    
    # Combine the detailed metrics
    combined_details = {
        'title_analysis': {
            'sentiment': title_sentiment,
            'confidence': title_confidence,
            'details': title_details
        },
        'description_analysis': {
            'sentiment': desc_sentiment,
            'confidence': desc_confidence,
            'details': desc_details
        }
    }
    
    # Calculate combined confidence
    combined_confidence = (title_confidence * title_weight + desc_confidence * desc_weight)
    
    # Determine final sentiment based on weighted confidences
    if title_details and desc_details:
        combined_score = (
            title_details['compound'] * title_weight +
            desc_details['compound'] * desc_weight
        )
        
        if combined_score >= 0.05:
            sentiment = "POSITIVE"
        elif combined_score <= -0.05:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
    else:
        sentiment = desc_sentiment  # fallback to description sentiment
    
    return sentiment, combined_confidence, combined_details

def get_tradingview_data(symbol):
    """Get stock data from TradingView"""
    
    # Format symbol for TradingView (add leading zeros if needed)
    symbol = str(symbol).zfill(4)  # Ensure 4 digits
    
    # TradingView API endpoint
    url = "https://scanner.tradingview.com/saudi/scan"
    
    # Request payload for Saudi market
    payload = {
        "filter": [{"left": "market_cap_basic", "operation": "nempty"}],
        "symbols": {"tickers": [f"TADAWUL:{symbol}"]},  # Removed .SR suffix
        "columns": [
            "close",
            "change",
            "volume", 
            "market_cap_basic",
            "price_earnings_ttm",
            "high",
            "low",
            "open",
            "Recommend.All",
            "RSI",
            "SMA20",
            "SMA50",
            "MACD.macd"
        ]
    }

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
        # Check if response is valid JSON
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON response from TradingView API")
            return None
        
        # Debug logging
        if 'data' not in data:
            st.error(f"Unexpected API response structure: {data}")
            return None
            
        if not data['data']:
            st.warning(f"No data found for symbol TADAWUL:{symbol}")
            return None
            
        stock_data = data['data'][0]['d']
        
        # Create DataFrame with safe value extraction
        df = pd.DataFrame({
            'Symbol': [symbol],
            'Close': [safe_get(stock_data, 0)],
            'Change %': [safe_get(stock_data, 1)],
            'Volume': [safe_get(stock_data, 2)],
            'Market Cap': [safe_get(stock_data, 3)],
            'P/E Ratio': [safe_get(stock_data, 4)],
            'High': [safe_get(stock_data, 5)],
            'Low': [safe_get(stock_data, 6)],
            'Open': [safe_get(stock_data, 7)],
            'Technical Rating': [safe_get(stock_data, 8)],
            'RSI': [safe_get(stock_data, 9)],
            'SMA20': [safe_get(stock_data, 10)],
            'SMA50': [safe_get(stock_data, 11)],
            'MACD': [safe_get(stock_data, 12)]
        })
        
        return df
            
    except requests.exceptions.RequestException as e:
        st.error(f"Network error when fetching data: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def safe_get(data_list, index, default=None):
    """Safely get value from list with default value if index error"""
    try:
        return data_list[index]
    except (IndexError, TypeError):
        return default

def get_technical_analysis(symbol):
    """Get technical analysis for Saudi stocks"""
    df = get_tradingview_data(symbol)
    
    if df is not None:
        analysis = {
            'close': df['Close'].iloc[0],
            'rsi': df['RSI'].iloc[0],
            'sma20': df['SMA20'].iloc[0],
            'sma50': df['SMA50'].iloc[0],
            'macd': df['MACD'].iloc[0],
            'technical_rating': df['Technical Rating'].iloc[0]
        }
        
        signals = []
        
        # RSI Analysis
        if analysis['rsi'] > 70:
            signals.append(("RSI", "Overbought", "SELL"))
        elif analysis['rsi'] < 30:
            signals.append(("RSI", "Oversold", "BUY"))
            
        # Moving Average Analysis
        if analysis['close'] > analysis['sma20'] > analysis['sma50']:
            signals.append(("Moving Averages", "Uptrend", "BUY"))
        elif analysis['close'] < analysis['sma20'] < analysis['sma50']:
            signals.append(("Moving Averages", "Downtrend", "SELL"))
            
        # MACD Analysis
        if analysis['macd'] > 0:
            signals.append(("MACD", "Bullish", "BUY"))
        elif analysis['macd'] < 0:
            signals.append(("MACD", "Bearish", "SELL"))
            
        # Overall Technical Rating
        signals.append(("Overall", f"Rating: {analysis['technical_rating']}", 
                      "BUY" if analysis['technical_rating'] > 0 else "SELL"))
        
        return signals
    return None

def plot_technical_chart(hist_data):
    """Create technical analysis chart using plotly"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=hist_data.index,
        open=hist_data['Open'],
        high=hist_data['High'],
        low=hist_data['Low'],
        close=hist_data['Close'],
        name='OHLC'
    ))
    
    # Add Moving Averages
    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['SMA20'], name='SMA20'))
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['BB_upper'], name='BB Upper',
                            line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['BB_lower'], name='BB Lower',
                            line=dict(dash='dash')))
    
    fig.update_layout(
        title='Price and Technical Indicators',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark'
    )
    
    return fig

def get_technical_signals(analysis):
    """Generate trading signals based on technical indicators for Saudi stocks"""
    signals = []
    
    # RSI Signals (adjusted for Tadawul volatility)
    if analysis['rsi'] > 75:  # More conservative overbought level
        signals.append(("RSI", "Strongly Overbought", "SELL"))
    elif analysis['rsi'] > 70:
        signals.append(("RSI", "Overbought", "SELL"))
    elif analysis['rsi'] < 25:  # More conservative oversold level
        signals.append(("RSI", "Strongly Oversold", "BUY"))
    elif analysis['rsi'] < 30:
        signals.append(("RSI", "Oversold", "BUY"))
        
    # MACD Signals
    if analysis['macd'] > analysis['macd_signal']:
        signals.append(("MACD", "Bullish Crossover", "BUY"))
    elif analysis['macd'] < analysis['macd_signal']:
        signals.append(("MACD", "Bearish Crossover", "SELL"))
        
    # Moving Average Signals
    if analysis['price'] > analysis['sma10'] > analysis['sma20']:
        signals.append(("MA", "Strong Bullish Trend", "BUY"))
    elif analysis['price'] < analysis['sma10'] < analysis['sma20']:
        signals.append(("MA", "Strong Bearish Trend", "SELL"))
        
    # Money Flow Index Signals
    if analysis['mfi'] > 80:
        signals.append(("MFI", "Overbought", "SELL"))
    elif analysis['mfi'] < 20:
        signals.append(("MFI", "Oversold", "BUY"))
        
    # Volume Analysis
    if analysis['volume'] > analysis['avg_volume_10d'] * 1.5:
        signals.append(("Volume", "Unusual High Volume", "WATCH"))
        
    return signals

def display_article(article):
    """Display a single news article with enhanced sentiment and technical analysis"""
    title = article.get("title", "No title available")
    description = article.get("description", "No description available")
    url = article.get("url", "#")
    published_at = article.get("published_at", "")
    source = article.get("source", "Unknown source")
    
    try:
        published_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%S.%fZ")
        published_str = published_date.strftime("%Y-%m-%d %H:%M")
    except:
        published_str = published_at

    with st.container():
        st.subheader(title)
        st.write(f"**Source:** {source} | **Published:** {published_str}")
        
        if description:
            st.write(description)
            sentiment, confidence, details = analyze_article_sentiment(article)
            
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            # Main sentiment metrics
            col1.metric("Overall Sentiment", sentiment)
            col2.metric("Confidence", f"{confidence:.2%}")
            
            # Detailed sentiment analysis
            with col3.expander("Detailed Analysis"):
                st.write("### Title Analysis")
                if details['title_analysis']['details']:
                    st.write(f"Sentiment: {details['title_analysis']['sentiment']}")
                    st.write(f"Confidence: {details['title_analysis']['confidence']:.2%}")
                
                st.write("### Description Analysis")
                if details['description_analysis']['details']:
                    st.write(f"Sentiment: {details['description_analysis']['sentiment']}")
                    st.write(f"Confidence: {details['description_analysis']['confidence']:.2%}")
                    
                    vader = details['description_analysis']['details']['vader']
                    finbert = details['description_analysis']['details']['finbert']
                    roberta = details['description_analysis']['details']['roberta']
                    
                    st.write("#### VADER Scores")
                    st.write(f"Positive: {vader['pos']:.2%}")
                    st.write(f"Neutral: {vader['neu']:.2%}")
                    st.write(f"Negative: {vader['neg']:.2%}")
                    
                    st.write("#### FinBERT Analysis")
                    st.write(f"Label: {finbert['label']}")
                    st.write(f"Score: {finbert['score']:.2%}")
                    
                    st.write("#### RoBERTa Analysis")
                    st.write(f"Label: {roberta['label']}")
                    st.write(f"Score: {roberta['score']:.2%}")
        
        # Add Technical Analysis section
        if 'entities' in article and article['entities']:
            for entity in article['entities']:
                if entity.get('type') == 'equity' and entity.get('symbol'):
                    st.write("### Technical Analysis")
                    with st.expander(f"Technical Analysis for {entity['name']} ({entity['symbol']})"):
                        analysis = get_technical_analysis(entity['symbol'])
                        
                        if analysis:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Current Price", f"SAR {analysis['price']:.2f}", 
                                        f"{analysis['change_percent']:.2f}%")
                                st.metric("RSI", f"{analysis['rsi']:.2f}")
                                st.metric("MACD", f"{analysis['macd']:.2f}")
                            
                            with col2:
                                st.metric("SMA20", f"SAR {analysis['sma20']:.2f}")
                                st.metric("Volume", f"{analysis['volume']:,.0f}")
                            
                            # Plot technical chart
                            st.plotly_chart(plot_technical_chart(analysis['historical_data']))
                            
                            # Display trading signals
                            signals = get_technical_signals(analysis)
                            if signals:
                                st.write("### Trading Signals")
                                for indicator, condition, signal in signals:
                                    st.write(f"- {indicator}: {condition} ({signal})")
                        else:
                            st.warning("Unable to fetch technical analysis data")
        
        st.markdown(f"[Read full article]({url})")
        st.markdown("---")

def main():
    st.title("Saudi Stock Market News")
    st.write("Real-time news analysis for Saudi stock market")

    # Sidebar configuration
    st.sidebar.title("Settings")
    max_limit = 3  # Set maximum limit based on API plan
    limit = st.sidebar.slider("Number of articles", 1, max_limit, max_limit)
    
    # Add plan information
    st.sidebar.info(f"Current plan limit: {max_limit} articles per request")
    
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
                    display_article(article)
            else:
                st.warning("No news articles found for the selected date range")

    # App information
    st.sidebar.markdown("---")
    st.sidebar.write("App Version: 1.0.1")
    if API_TOKEN:
        st.sidebar.success("API Token loaded successfully")
    else:
        st.sidebar.error("API Token not found")

if __name__ == "__main__":
    main()
