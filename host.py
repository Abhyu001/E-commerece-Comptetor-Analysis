#importing libraries
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


# Follow the steps in README.md file to get the API keys and Azure OpenAI credentials
API_KEY = "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" # Groq API Key
SLACK_WEBHOOK = "https://hooks.slack.com/services/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"

# Function to truncate long text to a specified length for better display
def truncate_text(text, max_length=512):
    return text[:max_length]

# Function to load competitor data from a CSV file
def load_competitor_data():
    """Load competitor data from a CSV file."""
    data = pd.read_csv("amazon_products.csv")
    print(data.head())  # Print first few rows of data for checking
    return data

# Function to load reviews data from a CSV file
def load_reviews_data():
    """Load reviews data from a CSV file."""
    reviews = pd.read_csv("review_data.csv")
    return reviews

# Function to apply sentiment analysis using the specific pre-trained model
def analyze_sentiment(reviews):
    """Analyze customer sentiment for reviews."""
    sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"
    )
    return sentiment_pipeline([str(review) for review in reviews])

# Function to train a predictive model using RandomForestRegressor to predict discount strategy
def train_predictive_model(data):
    """Train a predictive model for competitor pricing strategy."""
    # Cleaning and converting necessary columns for training
    data["Discount"] = data["Discount"].str.replace("%", "").astype(float)
    data["MRP Price"] = data["MRP Price"].astype(int)
    data["Predicted_Discount"] = data["Discount"] + (data["MRP Price"] * 0.05).round(2)

    X = data[["MRP Price", "Discount"]]  # Independent features for training
    y = data["Predicted_Discount"]  # Dependent target variable (Predicted Discount)
    print(X)  # Printing feature set to check
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, train_size=0.8
    )

    # Model initialization and training
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

import numpy as np
import pandas as pd

# Function to fit an ARIMA model for forecasting future discounts
# ARIMA(p,d,q) where p is autoregressive part, d ensures stationarity, q is moving average part
def forecast_discounts_arima(data, future_days=5):
    """
    Forecast future discounts using ARIMA.
    :param data: DataFrame containing historical discount data (with a datetime index).
    :param future_days: Number of days to forecast.
    :return: DataFrame with historical and forecasted discounts.
    """

    data = data.sort_index()  # Sort data by date index for time series analysis
    print(product_data.index)

    data["Discount"] = pd.to_numeric(data["Discount"], errors="coerce")  # Convert to numeric values
    data = data.dropna(subset=["Discount"])  # Remove rows with missing discount values

    discount_series = data["Discount"]
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)  # Convert index to datetime if not already
        except Exception as e:
            raise ValueError(
                "Index must be datetime or convertible to datetime."
            ) from e
    # ARIMA(p,d,q) model fitting
    model = ARIMA(discount_series, order=(5, 1, 0))  # ARIMA model configuration
    model_fit = model.fit()  # Fit the ARIMA model

    forecast = model_fit.forecast(steps=future_days)  # Forecast for future days
    future_dates = pd.date_range(
        start=discount_series.index[-1] + pd.Timedelta(days=1), periods=future_days
    )

    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast})  # Store predictions
    forecast_df.set_index("Date", inplace=True)  # Set the date as index

    return forecast_df

# Function to send data (recommendations) to Slack using webhook
def send_to_slack(data):
    """Send message data to Slack using webhook."""
    payload = {"text": data}
    response = requests.post(
        SLACK_WEBHOOK,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )

# Function to generate strategic business recommendations using AI model
def generate_strategy_recommendation(product_name, competitor_data, sentiment):
    """Generate strategic recommendations using an LLM (Large Language Model)."""
    date = datetime.now()
    prompt = f"""
    You are a highly skilled business strategist specializing in e-commerce. Based on the following details, suggest actionable strategies to optimize pricing, promotions, and customer satisfaction for the selected product:

1. Product Name: {product_name}

2. Competitor Data (including current prices, discounts, and predicted discounts):
{competitor_data}

3. Sentiment Analysis:
{sentiment}

5. Today's Date: {str(date)}

### Task:
- Analyze the competitor data and identify key pricing trends.
- Leverage sentiment analysis insights to highlight areas where customer satisfaction can be improved.
- Use the discount predictions to suggest how pricing strategies can be optimized over the next 5 days.
- Recommend promotional campaigns or marketing strategies that align with customer sentiments and competitive trends.
- Ensure the strategies are actionable, realistic, and geared toward increasing customer satisfaction, driving sales, and outperforming competitors.

Provide your recommendations in a structured format:
1. Pricing Strategy
2. Promotional Campaign Ideas
3. Customer Satisfaction Recommendations
    """

    messages = [{"role": "user", "content": prompt}]

    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama3-8b-8192",
        "temperature": 0,
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    # Sending the request to AI model and getting response
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(data),
        headers=headers,
    )
    res = res.json()
    response = res["choices"][0]["message"]["content"]
    return response


####--------------------------------------------------##########
# Streamlit app title and page configuration
st.set_page_config(page_title="E-Commerce Competitor Strategy Dashboard", layout="wide")

# Title of the dashboard and sidebar options
st.title("E-Commerce Competitor Strategy Dashboard")
st.sidebar.header("Select a Product")

# List of products to choose from in the sidebar
products = [
    "boAt Rockerz 480 w/RGB LEDs, 6 Light Modes, 40mm Drivers, Beast Mode, 60hrs Playback, ENx Tech, BT v5.3, Adaptive Fit & Easy Access Controls, Bluetooth Headphones(Black Sabre)",
    "HP Victus, 13th Gen Intel Core i5-13420H, 6GB NVIDIA RTX 4050, 16GB DDR4, 512GB SSD (Win11, Office 21, Silver, 2.29kg) 144Hz, 9MS, IPS, 15.6-inch(39.6cm) FHD Gaming Laptop, Enhanced Cooling, fa1319TX",
    "HAVAI Thunder 85 Desert Cooler - 75 Litres, 16 Inch Blade, Black",
    "Samsung Galaxy M05 (Mint Green, 4GB RAM, 64 GB Storage) | 50MP Dual Camera | Bigger 6.7 HD+ Display | 5000mAh Battery | 25W Fast Charging | 2 Gen OS Upgrade & 4 Year Security Update | Without Charger",
    "iQOO Z9x 5G (Tornado Green, 6GB RAM, 128GB Storage) | Snapdragon 6 Gen 1 with 560k+ AnTuTu Score | 6000mAh Battery with 7.99mm Slim Design | 44W FlashCharge"
]
selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)

# Calling functions part of Driver Code
competitor_data = load_competitor_data()  # Load competitor data
reviews_data = load_reviews_data()  # Load customer reviews data

# Filter data for selected product
product_data = competitor_data[competitor_data["Title"] == selected_product]
product_reviews = reviews_data[reviews_data["Title"] == selected_product]

# Display Competitor Data for selected product
st.header(f"Competitor Analysis for {selected_product}")
st.subheader("Competitor Data")
st.table(product_data.tail(5))

# Perform Sentiment Analysis on Reviews
if not product_reviews.empty:
    product_reviews["Reviews"] = product_reviews["One_Review"].apply(
        lambda x: truncate_text(x, 512)  # Truncate long reviews
    )
    reviews = product_reviews["Reviews"].tolist()
    sentiments = analyze_sentiment(reviews)  # Get sentiment results

    # Display Sentiment Analysis Results
    st.subheader("Customer Sentiment Analysis")
    sentiment_df = pd.DataFrame(sentiments)
    fig = px.bar(sentiment_df, x="label", title="Sentiment Analysis Results")
    st.plotly_chart(fig)
else:
    st.write("No reviews available for this product.")  # No reviews available message

# Preprocessing the competitor data (date column conversion, cleaning)
product_data["Date"] = pd.to_datetime(product_data["Date"], errors="coerce")
product_data = product_data.dropna(subset=["Date"])  # Drop rows with missing dates
product_data.set_index("Date", inplace=True)  # Set Date as index
product_data = product_data.sort_index()  # Sort by date index

# Convert discount column to numeric and drop rows with missing discounts
product_data["Discount"] = pd.to_numeric(product_data["Discount"], errors="coerce")
product_data = product_data.dropna(subset=["Discount"])

# Forecasting Model to predict future discounts
product_data_with_predictions = forecast_discounts_arima(product_data)

# Display Competitor's current and predicted discounts
st.subheader("Competitor Current and Predicted Discounts")
st.table(product_data_with_predictions.tail(10))

# Generate and display strategic recommendations
recommendations = generate_strategy_recommendation(
    selected_product,
    product_data_with_predictions,
    sentiments if not product_reviews.empty else "No reviews available",  # Handle missing reviews
)
st.subheader("Strategic Recommendations")
st.write(recommendations)

# Send recommendations to Slack
send_to_slack(recommendations)
