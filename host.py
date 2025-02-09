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
    page_title="E-Commerce Competitor Strategy Dashboard",  # Set the page title
    layout="wide"  # Use wide layout for better UI presentation
)

# API Keys (Ensure these are stored securely)
API_KEY = "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"  # Groq API Key for AI model access
SLACK_WEBHOOK = "https://hooks.slack.com/services/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"  # Slack webhook for sending notifications

# Function to truncate long text for better display
def truncate_text(text, max_length=512):
    return text[:max_length]  # Truncate reviews to 512 characters to avoid display issues

# Load competitor data from CSV file
@st.cache_data  # Caching the data for performance optimization
def load_competitor_data():
    data = pd.read_csv("amazon_products.csv")  # Load product data from CSV
    return data

# Load customer reviews data from CSV file
@st.cache_data  # Caching to prevent reloading the data each time
def load_reviews_data():
    reviews = pd.read_csv("review_data.csv")  # Load customer reviews data from CSV
    return reviews

# Perform sentiment analysis on reviews
@st.cache_data  # Cache the sentiment analysis results to avoid re-running the analysis
def analyze_sentiment(reviews):
    sentiment_pipeline = pipeline(  # Load pre-trained sentiment analysis model
        "sentiment-analysis", 
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0f"
    )
    return sentiment_pipeline([str(review) for review in reviews])  # Analyze sentiment for each review

# Train a predictive pricing model using RandomForest
@st.cache_data  # Cache the model to avoid re-training every time
def train_predictive_model(data):
    data["Discount"] = data["Discount"].str.replace("%", "").astype(float)  # Clean and convert discount to float
    data["MRP Price"] = data["MRP Price"].astype(int)  # Convert MRP price to integer
    data["Predicted_Discount"] = data["Discount"] + (data["MRP Price"] * 0.05).round(2)  # Predict discount based on MRP

    X = data[["MRP Price", "Discount"]]  # Features for the model (MRP price and current discount)
    y = data["Predicted_Discount"]  # Target (predicted discount)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data for training and testing
    model = RandomForestRegressor(random_state=42)  # Initialize the RandomForest model
    model.fit(X_train, y_train)  # Train the model
    return model

# Function to forecast future discounts using ARIMA model
@st.cache_data  # Cache the ARIMA forecast to avoid recalculating every time
def forecast_discounts_arima(data, future_days=5):
    data = data.sort_index()  # Ensure data is sorted by date
    data["Discount"] = pd.to_numeric(data["Discount"], errors="coerce")  # Convert discount column to numeric
    data = data.dropna(subset=["Discount"])  # Drop rows with missing discount values

    discount_series = data["Discount"]  # Select the discount column for forecasting
    data.index = pd.to_datetime(data.index, errors="coerce")  # Ensure the index is a datetime type

    model = ARIMA(discount_series, order=(5, 1, 0))  # ARIMA model with (p=5, d=1, q=0)
    model_fit = model.fit()  # Fit the ARIMA model
    forecast = model_fit.forecast(steps=future_days)  # Forecast future discount values for the specified number of days
    
    # Create future dates for the forecasted period
    future_dates = pd.date_range(start=discount_series.index[-1] + pd.Timedelta(days=1), periods=future_days)
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast})  # Prepare the forecast DataFrame
    forecast_df.set_index("Date", inplace=True)  # Set the date column as the index
    return forecast_df

# Send recommendations to Slack
@st.cache_data  # Cache Slack notifications to optimize performance
def send_to_slack(data):
    payload = {"text": data}  # Format the data into a Slack message
    requests.post(
        SLACK_WEBHOOK,  # Slack Webhook URL to send the message
        data=json.dumps(payload),  # Send the payload as JSON
        headers={"Content-Type": "application/json"},  # Set content type to JSON
    )

# Generate business strategy recommendations using AI
@st.cache_data  # Cache the strategy recommendations for performance
def generate_strategy_recommendation(product_name, competitor_data, sentiment):
    date = datetime.now()  # Get current date and time for context
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
    res = res.json()  # Parse the response from the AI model
    return res["choices"][0]["message"]["content"]  # Return the strategy recommendations from the AI model

# Streamlit UI Setup
st.title("📊 E-Commerce Competitor Strategy Dashboard")  # Title for the main page
st.sidebar.header("Select a Product")  # Sidebar header for selecting a product

# List of products available for analysis
products = [
    "boAt Rockerz 480", "HP Victus Gaming Laptop", "HAVAI Thunder 85 Cooler", "Samsung Galaxy M05", "iQOO Z9x 5G"
]
selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)  # Dropdown menu for product selection

# Load data
competitor_data = load_competitor_data()  # Load competitor data
reviews_data = load_reviews_data()  # Load reviews data

# Filter data for the selected product
product_data = competitor_data[competitor_data["Title"] == selected_product]  # Filter competitor data for selected product
product_reviews = reviews_data[reviews_data["Title"] == selected_product]  # Filter reviews data for selected product

# Display Competitor Data
st.header(f"Competitor Analysis for {selected_product}")  # Header for competitor analysis section
st.subheader("Competitor Data")  # Subheader for competitor data table
st.dataframe(product_data.tail(5), width=800)  # Display the last 5 rows of the competitor data in a table

# Sentiment Analysis
if not product_reviews.empty:  # Check if reviews are available for the product
    product_reviews["Reviews"] = product_reviews["Reviews"].apply(lambda x: truncate_text(x, 512))  # Truncate reviews
    reviews = product_reviews["Reviews"].tolist()  # Convert reviews to a list
    sentiments = analyze_sentiment(reviews)  # Perform sentiment analysis on reviews
    
    st.subheader("Customer Sentiment Analysis")  # Header for sentiment analysis section
    sentiment_df = pd.DataFrame(sentiments)  # Convert sentiment analysis results into a DataFrame
    fig = px.bar(sentiment_df, x="label", title="Sentiment Analysis Results")  # Create a bar chart for sentiment analysis
    st.plotly_chart(fig)  # Display the sentiment analysis chart
else:
    st.write("No reviews available for this product.")  # Message if no reviews are available

# Generate Recommendations
recommendations = generate_strategy_recommendation(selected_product, product_data, sentiments if not product_reviews.empty else "No reviews available")  # Generate strategy recommendations based on product data and sentiment
st.subheader("Strategic Recommendations")  # Header for recommendations section
st.write(recommendations)  # Display the generated recommendations
send_to_slack(recommendations)  # Send the recommendations to Slack for further action
