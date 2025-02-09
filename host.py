import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from openai import AzureOpenAI
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline
import numpy as np

# Streamlit Page Configuration for Better UI Experience
st.set_page_config(
    page_title="E-Commerce Competitor Strategy Dashboard",
    layout="wide"
)

# API Keys (Ensure these are stored securely)
API_KEY = "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"  # Groq API Key
SLACK_WEBHOOK = "https://hooks.slack.com/services/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"

# Function to truncate long text for better display
def truncate_text(text, max_length=512):
    return text[:max_length]

# Load competitor data from CSV file
@st.cache_data  # Caching to optimize performance
def load_competitor_data():
    data = pd.read_csv("amazon_products.csv")
    return data

# Load customer reviews data from CSV file
@st.cache_data
def load_reviews_data():
    reviews = pd.read_csv("review_data.csv")
    return reviews

# Perform sentiment analysis on reviews
@st.cache_data
def analyze_sentiment(reviews):
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0f"
    )
    return sentiment_pipeline([str(review) for review in reviews])

# Train a predictive pricing model using RandomForest
@st.cache_data
def train_predictive_model(data):
    data["Discount"] = data["Discount"].str.replace("%", "").astype(float)
    data["MRP Price"] = data["MRP Price"].astype(int)
    data["Predicted_Discount"] = data["Discount"] + (data["MRP Price"] * 0.05).round(2)

    X = data[["MRP Price", "Discount"]]
    y = data["Predicted_Discount"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to forecast future discounts using ARIMA model
@st.cache_data
def forecast_discounts_arima(data, future_days=5):
    data = data.sort_index()
    data["Discount"] = pd.to_numeric(data["Discount"], errors="coerce")
    data = data.dropna(subset=["Discount"])

    discount_series = data["Discount"]
    data.index = pd.to_datetime(data.index, errors="coerce")

    model = ARIMA(discount_series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=future_days)
    
    future_dates = pd.date_range(start=discount_series.index[-1] + pd.Timedelta(days=1), periods=future_days)
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast})
    forecast_df.set_index("Date", inplace=True)
    return forecast_df

# Send recommendations to Slack
@st.cache_data
def send_to_slack(data):
    payload = {"text": data}
    requests.post(
        SLACK_WEBHOOK,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )

# Generate business strategy recommendations using AI
@st.cache_data
def generate_strategy_recommendation(product_name, competitor_data, sentiment):
    date = datetime.now()
    prompt = f"""
    You are a business strategist specializing in e-commerce. Based on the details below, suggest strategies:
    
    1. Product: {product_name}
    2. Competitor Data:
    {competitor_data}
    3. Sentiment Analysis:
    {sentiment}
    4. Date: {str(date)}
    
    Provide structured recommendations:
    1. Pricing Strategy
    2. Promotional Campaign Ideas
    3. Customer Satisfaction Improvements
    """

    data = {"messages": [{"role": "user", "content": prompt}], "model": "llama3-8b-8192", "temperature": 0}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    res = requests.post("https://api.groq.com/openai/v1/chat/completions", data=json.dumps(data), headers=headers)
    res = res.json()
    return res["choices"][0]["message"]["content"]

# Streamlit UI Setup
st.title("📊 E-Commerce Competitor Strategy Dashboard")
st.sidebar.header("Select a Product")

products = [
    "boAt Rockerz 480", "HP Victus Gaming Laptop", "HAVAI Thunder 85 Cooler", "Samsung Galaxy M05", "iQOO Z9x 5G"
]
selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)

# Load data
competitor_data = load_competitor_data()
reviews_data = load_reviews_data()

# Filter data for selected product
product_data = competitor_data[competitor_data["Title"] == selected_product]
product_reviews = reviews_data[reviews_data["Title"] == selected_product]

# Display Competitor Data
st.header(f"Competitor Analysis for {selected_product}")
st.subheader("Competitor Data")
st.table(product_data.tail(5))

# Sentiment Analysis
if not product_reviews.empty:
    product_reviews["Reviews"] = product_reviews["Reviews"].apply(lambda x: truncate_text(x, 512))
    reviews = product_reviews["Reviews"].tolist()
    sentiments = analyze_sentiment(reviews)
    
    st.subheader("Customer Sentiment Analysis")
    sentiment_df = pd.DataFrame(sentiments)
    fig = px.bar(sentiment_df, x="label", title="Sentiment Analysis Results")
    st.plotly_chart(fig)
else:
    st.write("No reviews available for this product.")

# Forecasting Future Discounts
st.subheader("Competitor Current and Predicted Discounts")
st.dataframe(product_data_with_predictions.tail(10), width=800)  # Adjusting table width for better readability

# Generate Recommendations
recommendations = generate_strategy_recommendation(selected_product, product_data_with_predictions, sentiments if not product_reviews.empty else "No reviews available")
st.subheader("Strategic Recommendations")
st.write(recommendations)
send_to_slack(recommendations)
