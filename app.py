import streamlit as st
import requests

# API Configuration
NEWS_API_URL = "https://api.marketaux.com/v1/news/all"
API_TOKEN = st.secrets["general"]["MARKETAUX_API_KEY"]

def fetch_news():
    """Test API connection"""
    params = {
        "api_token": API_TOKEN,
        "countries": "sa",
        "filter_entities": "true",
        "limit": 1
    }
    
    try:
        st.write("Attempting to connect to API...")
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data:
            st.error(f"API Error: {data['error']['message']}")
            return
            
        st.success("API connection successful!")
        st.write("Sample response:", data)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

def main():
    st.title("API Test")
    
    if st.button("Test API Connection"):
        fetch_news()

if __name__ == "__main__":
    main()
