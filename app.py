import requests
import streamlit as st
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

# API Configuration
NEWS_API_URL = "https://api.marketaux.com/v1/news/all"
API_TOKEN = st.secrets["STOCKDATA_API_TOKEN"]

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
    """Fetch news articles from the API"""
    params = {
        "countries": "sa",
        "filter_entities": "true",
        "limit": limit,
        "published_after": published_after,
        "api_token": API_TOKEN
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        st.error(f"Error fetching news: {e}")
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

def display_article(article):
    """Display a single news article with enhanced sentiment analysis"""
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
        
        st.markdown(f"[Read full article]({url})")
        st.markdown("---")

def main():
    st.title("Saudi Stock Market News")
    st.write("Real-time news analysis for Saudi stock market")

    # Sidebar configuration
    st.sidebar.title("Settings")
    limit = st.sidebar.slider("Number of articles", 1, 20, 10)
    
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
