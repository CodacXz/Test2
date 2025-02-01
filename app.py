import requests
import streamlit as st
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import ta  # Technical Analysis library
import plotly.graph_objects as go
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
MARKETSTACK_API_KEY = os.getenv("MARKETSTACK_API_KEY")

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

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

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

def get_fmp_data(symbol):
    """Get stock data from Financial Modeling Prep for Saudi stocks"""
    API_KEY = FMP_API_KEY
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    # Format symbol for Saudi stocks
    formatted_symbol = f"{symbol}.SAU"
    
    try:
        # Get real-time quote
        quote_url = f"{BASE_URL}/quote/{formatted_symbol}?apikey={API_KEY}"
        response = requests.get(quote_url)
        data = response.json()
        
        if data and len(data) > 0:
            quote = data[0]
            
            # Get company profile for additional info
            profile_url = f"{BASE_URL}/profile/{formatted_symbol}?apikey={API_KEY}"
            profile_response = requests.get(profile_url)
            profile_data = profile_response.json()
            company_profile = profile_data[0] if profile_data else {}
            
            # Get technical indicators
            tech_url = f"{BASE_URL}/technical_indicator/daily/{formatted_symbol}?period=14&type=rsi,sma&apikey={API_KEY}"
            tech_response = requests.get(tech_url)
            tech_data = tech_response.json()
            
            return {
                'symbol': symbol,
                'name': company_profile.get('companyName', 'N/A'),
                'sector': company_profile.get('sector', 'N/A'),
                'price': quote.get('price', 0),
                'change_percent': quote.get('changesPercentage', 0),
                'volume': quote.get('volume', 0),
                'market_cap': quote.get('marketCap', 0),
                'pe_ratio': quote.get('pe', 0),
                'eps': quote.get('eps', 0),
                'high': quote.get('dayHigh', 0),
                'low': quote.get('dayLow', 0),
                'open': quote.get('open', 0),
                'previous_close': quote.get('previousClose', 0),
                'year_high': quote.get('yearHigh', 0),
                'year_low': quote.get('yearLow', 0),
                'rsi': tech_data[0].get('rsi', 0) if tech_data else 0,
                'sma20': tech_data[0].get('sma', 0) if tech_data else 0
            }
            
        elif "Error Message" in data:
            st.error(f"API Error: {data['Error Message']}")
        return None
            
        return None
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def get_marketstack_data(symbol):
    """Get stock data from Marketstack for Saudi stocks"""
    API_KEY = "e537111ec25329587f27f61bb59938bc"
    BASE_URL = "http://api.marketstack.com/v1"
    
    # Format symbol for Saudi stocks (remove leading zeros and add exchange)
    symbol = str(int(symbol))  # Remove leading zeros
    formatted_symbol = f"TADAWUL:{symbol}"  # Use TADAWUL exchange prefix
    
    try:
        # Get real-time data
        params = {
            "access_key": API_KEY,
            "symbols": formatted_symbol,
            "limit": 1,
            "exchange": "TADAWUL"  # Specify Saudi exchange
        }
        
        # Get latest quote
        response = requests.get(f"{BASE_URL}/intraday/latest", params=params)
        data = response.json()
        
        if "data" in data and len(data["data"]) > 0:
            quote = data["data"][0]
            
            # Get EOD data for technical analysis
            eod_params = {
                "access_key": API_KEY,
                "symbols": formatted_symbol,
                "exchange": "TADAWUL",
                "limit": 30  # Get last 30 days for technical analysis
            }
            
            eod_response = requests.get(f"{BASE_URL}/eod", params=eod_params)
            eod_data = eod_response.json()
            
            # Calculate technical indicators
            if "data" in eod_data and len(eod_data["data"]) > 0:
                df = pd.DataFrame(eod_data["data"])
                df['close'] = pd.to_numeric(df['close'])
                
                rsi = ta.momentum.RSIIndicator(df['close']).rsi().iloc[-1]
                sma20 = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator().iloc[-1]
                
                return {
                    'symbol': symbol,
                    'price': quote.get('last', 0),
                    'open': quote.get('open', 0),
                    'high': quote.get('high', 0),
                    'low': quote.get('low', 0),
                    'volume': quote.get('volume', 0),
                    'change': quote.get('change', 0),
                    'change_percent': quote.get('change_percent', 0),
                    'last_updated': quote.get('date', ''),
                    'rsi': rsi,
                    'sma20': sma20,
                    'historical_data': df  # Add historical data for charts
                }
            
        if "error" in data:
            st.error(f"API Error: {data['error']['message']}")
            return None
            
        return None
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def get_technical_analysis(symbol):
    """Get technical analysis using Marketstack data"""
    data = get_marketstack_data(symbol)
    
    if data:
        signals = []
        
        # RSI Analysis
        rsi = data.get('rsi')
        if rsi:
            if rsi > 70:
                signals.append(("RSI", "Overbought", "SELL"))
            elif rsi < 30:
                signals.append(("RSI", "Oversold", "BUY"))
        
        # Moving Average Analysis
        price = data.get('price')
        sma20 = data.get('sma20')
        if price and sma20:
            if price > sma20:
                signals.append(("SMA20", "Above Average", "BUY"))
            else:
                signals.append(("SMA20", "Below Average", "SELL"))
        
        # Price Change Analysis
        change = data.get('change_percent')
        if change:
            if change > 5:
                signals.append(("Price Change", "Strong Momentum", "BUY"))
            elif change < -5:
                signals.append(("Price Change", "Weak Momentum", "SELL"))
        
        return {
            'price': data['price'],
            'change_percent': data['change_percent'],
            'volume': data['volume'],
            'signals': signals,
            'rsi': data['rsi'],
            'sma20': data['sma20'],
            'last_updated': data['last_updated']
        }
    
    return None

def get_technical_signals(analysis):
    """Generate trading signals based on technical indicators"""
    signals = []
    
    if not analysis:
        return signals
        
    # RSI Analysis
    rsi = analysis.get('rsi')
    if rsi:
        if rsi > 70:
            signals.append(("RSI", "Overbought", "SELL"))
        elif rsi < 30:
            signals.append(("RSI", "Oversold", "BUY"))
            
    # Moving Average Analysis
    price = analysis.get('price')
    sma20 = analysis.get('sma20')
    if all(v is not None for v in [price, sma20]):
        if price > sma20:
            signals.append(("SMA20", "Above Average", "BUY"))
        else:
            signals.append(("SMA20", "Below Average", "SELL"))
            
    # Price Change Analysis
    change = analysis.get('change_percent')
    if change:
        if change > 5:
            signals.append(("Price Change", "Strong Momentum", "BUY"))
        elif change < -5:
            signals.append(("Price Change", "Weak Momentum", "SELL"))
        
    return signals

def get_symbol_from_name(company_name):
    """Map company name to stock symbol"""
    # Common company name variations to symbol mapping
    company_map = {
        # Banks
        "al rajhi": "1120",
        "alrajhi": "1120",
        "sns": "1180",
        "saudi national bank": "1180",
        "alinma": "1150",
        "al inma": "1150",
        
        # Energy & Materials
        "aramco": "2222",
        "saudi aramco": "2222",
        "sabic": "2010",
        "saudi basic industries": "2010",
        "maaden": "1211",
        "saudi arabian mining": "1211",
        
        # Healthcare
        "canadian medical": "9518",
        "canadian medical center": "9518",
        "dallah health": "4004",
        "mouwasat": "4002",
        
        # Add more mappings as needed
    }
    
    # Clean and normalize the company name
    name = company_name.lower().strip()
    
    # Try exact match first
    if name in company_map:
        return company_map[name]
    
    # Try partial matches
    for key, symbol in company_map.items():
        if key in name or name in key:
            return symbol
    
    return None

def display_article(article):
    """Display news article with sentiment and technical analysis"""
    title = article.get("title", "")
    description = article.get("description", "")
    source = article.get("source", "")
    url = article.get("url", "#")
    published_at = article.get("published_at", "")
    
    # Extract company name and get symbol
    company_name = extract_company_name(title + " " + description)
    if company_name:
        symbol = get_symbol_from_name(company_name)
    else:
        symbol = None
    
    # Display article info
    st.markdown(f"### {title}")
    st.write(f"Source: {source}")
    st.write(f"Published: {published_at}")
    
    if description:
        st.write(description)
    
    # Display sentiment analysis
    sentiment, confidence, details = analyze_article_sentiment(article)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Sentiment Analysis")
        st.write(f"Overall: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}")
    
    # If we found a company symbol, show technical analysis
    if symbol:
        with col2:
            st.write("### Technical Analysis")
            st.write(f"Company: {company_name} ({symbol})")
            
            # Get technical analysis
            analysis = get_marketstack_data(symbol)
            if analysis:
                # Show price and change
                st.metric("Price", 
                         f"SAR {analysis['price']:.2f}", 
                         f"{analysis['change_percent']:.2f}%")
                
                # Show technical indicators
                st.write(f"RSI: {analysis['rsi']:.2f}")
                st.write(f"SMA20: {analysis['sma20']:.2f}")
                
                # Plot chart if historical data is available
                if 'historical_data' in analysis:
                    st.plotly_chart(plot_technical_chart(analysis['historical_data']))
                
                # Display trading signals
                signals = get_technical_signals(analysis)
                if signals:
                    st.write("### Trading Signals")
                    for indicator, condition, signal in signals:
                        st.write(f"- {indicator}: {condition} ({signal})")
    
    st.markdown(f"[Read full article]({url})")
    st.markdown("---")

def extract_company_name(text):
    """Extract company name from text using simple rules"""
    # List of common company indicators
    indicators = [
        "company", "corp", "corporation", "inc", "ltd", "limited",
        "bank", "group", "holding", "شركة", "بنك", "مجموعة"
    ]
    
    # Split text into sentences
    sentences = text.split(".")
    
    for sentence in sentences:
        # Look for company indicators
        for indicator in indicators:
            if indicator.lower() in sentence.lower():
                # Get the surrounding words
                words = sentence.split()
                idx = next((i for i, word in enumerate(words) 
                          if indicator.lower() in word.lower()), None)
                if idx is not None:
                    # Take up to 3 words before the indicator
                    start = max(0, idx - 3)
                    company = " ".join(words[start:idx+1])
                    return company.strip()
    
    return None

def get_alpha_vantage_data(symbol):
    """Get stock data from Alpha Vantage with enhanced Saudi market support"""
    try:
        # Get quote data
        quote_data = get_alpha_vantage_quote(symbol)
        if quote_data is None:
            return None
            
        # Get technical indicators
        tech_data = get_alpha_vantage_technical(symbol)
        if tech_data:
            quote_data.update(tech_data)
            
        return pd.DataFrame([quote_data])
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def get_alpha_vantage_quote(symbol):
    """Get real-time quote from Alpha Vantage"""
    BASE_URL = "https://www.alphavantage.co/query"
    
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": f"{symbol}.SAU",  # Saudi stock format
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if "Global Quote" in data and data["Global Quote"]:
            quote = data["Global Quote"]
            return {
                'Symbol': symbol,
                'Close': float(quote.get('05. price', 0)),
                'Change %': float(quote.get('10. change percent', '0').replace('%', '')),
                'Volume': int(quote.get('06. volume', 0)),
                'Open': float(quote.get('02. open', 0)),
                'High': float(quote.get('03. high', 0)),
                'Low': float(quote.get('04. low', 0))
            }
            
        if "Note" in data:
            st.warning("API rate limit reached. Please wait a minute before trying again.")
            return None
            
        return None
        
    except Exception as e:
        st.error(f"Error fetching quote: {str(e)}")
        return None

def get_alpha_vantage_technical(symbol):
    """Get technical indicators from Alpha Vantage"""
    BASE_URL = "https://www.alphavantage.co/query"
    
    # Get RSI
    params = {
        "function": "RSI",
        "symbol": f"{symbol}.SAU",
        "interval": "daily",
        "time_period": 14,
        "series_type": "close",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if "Technical Analysis: RSI" in data:
            latest_date = max(data["Technical Analysis: RSI"].keys())
            rsi = float(data["Technical Analysis: RSI"][latest_date]["RSI"])
            
            # Get SMA
            sma_params = {
                "function": "SMA",
                "symbol": f"{symbol}.SAU",
                "interval": "daily",
                "time_period": 20,
                "series_type": "close",
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            
            sma_response = requests.get(BASE_URL, params=sma_params)
            sma_data = sma_response.json()
            
            if "Technical Analysis: SMA" in sma_data:
                latest_sma_date = max(sma_data["Technical Analysis: SMA"].keys())
                sma = float(sma_data["Technical Analysis: SMA"][latest_sma_date]["SMA"])
                
                return {
                    'RSI': rsi,
                    'SMA20': sma
                }
                
        if "Note" in data:
            st.warning("API rate limit reached. Please wait a minute before trying again.")
            return None
            
        return None
        
    except Exception as e:
        st.error(f"Error fetching technical indicators: {str(e)}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_alpha_vantage_daily(symbol):
    """Get daily historical data from Alpha Vantage"""
    BASE_URL = "https://www.alphavantage.co/query"
    
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": f"{symbol}.SAU",
        "outputsize": "compact",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if "Time Series (Daily)" in data:
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            
            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            return df
            
        if "Note" in data:
            st.warning("API rate limit reached. Please wait a minute before trying again.")
            return None
            
        return None
        
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return None

def get_fmp_technical_indicators(symbol):
    """Get technical indicators from FMP"""
    API_KEY = FMP_API_KEY
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    try:
        # Get technical indicators
        tech_url = f"{BASE_URL}/technical_indicator/daily/{symbol}.SAU?period=10&type=rsi,sma,ema,macd&apikey={API_KEY}"
        response = requests.get(tech_url)
        data = response.json()
        
        if data and len(data) > 0:
            latest = data[0]
            return {
                'RSI': latest.get('rsi', 0),
                'SMA20': latest.get('sma', 0),
                'EMA': latest.get('ema', 0),
                'MACD': latest.get('macd', 0),
                'MACD_Signal': latest.get('signal', 0)
            }
            
        return None
        
    except Exception as e:
        st.error(f"Error fetching technical indicators: {str(e)}")
        return None

def plot_technical_chart(df):
    """Create technical analysis chart using plotly"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))
    
    # Add SMA20
    sma20 = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=sma20,
        name='SMA20',
        line=dict(color='orange')
    ))
    
    # Update layout
    fig.update_layout(
        title='Price and Technical Indicators',
        yaxis_title='Price (SAR)',
        xaxis_title='Date',
        template='plotly_white'
    )
    
    return fig

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
