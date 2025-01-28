import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline

# API and Webhook placeholders
API_KEY = "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" #Groq API Key
SLACK_WEBHOOK = "https://hooks.slack.com/services/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"


def truncate_text(text, max_length=512):
    return text[:max_length]

def load_price_data():
    """Load price data from a CSV file."""
    data = pd.read_csv("/content/amazon_price.csv")
    data["MRP Price"] = data["MRP Price"].astype(str).str.replace(",", "").astype(float)
    data["Price"] = data["Price"].str.replace(",", "").astype(float)
    data["Discount (%)"] = data["Discount (%)"].astype(float)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    return data

def load_reviews_data():
    """Load reviews data from a CSV file."""
    reviews = pd.read_csv("/content/amazon_reviews.csv")
    return reviews

def analyze_sentiment(reviews):
    """Analyze customer sentiment for reviews."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    return sentiment_pipeline(reviews)

def train_predictive_model(data):
    """Train a predictive model for competitor pricing strategy."""
    X = data[["Price", "Discount (%)"]]
    y = data["MRP Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, train_size=0.8
    )
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def forecast_discounts_arima(data, future_days=5):
    """
    Forecast future discounts using ARIMA.
    :param data: DataFrame containing historical discount data (with a datetime index).
    :param future_days: Number of days to forecast.
    :return: DataFrame with historical and forecasted discounts.
    """
    data = data.sort_index()
    data = data.dropna(subset=["Discount (%)"])
    discount_series = data["Discount (%)"]

    model = ARIMA(discount_series, order=(5, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=future_days)
    future_dates = pd.date_range(
        start=discount_series.index[-1] + pd.Timedelta(days=1), periods=future_days
    )

    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount (%)": forecast})
    forecast_df.set_index("Date", inplace=True)

    return forecast_df

def send_to_slack(data):
    """Send recommendations to Slack."""
    payload = {"text": data}
    requests.post(
        SLACK_WEBHOOK,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )

def generate_strategy_recommendation(product_name, competitor_data, sentiment):
    """Generate strategic recommendations using an LLM."""
    date = datetime.now()
    prompt = f"""
    You are a highly skilled business strategist specializing in e-commerce. Based on the following details, suggest actionable strategies to optimize pricing, promotions, and customer satisfaction for the selected product:

1. **Product Name**: {product_name}

2. **Competitor Data** (including current prices, discounts, and predicted discounts):
{competitor_data}

3. **Sentiment Analysis**:
{sentiment}

5. **Today's Date**: {str(date)}

### Task:
- Analyze the competitor data and identify key pricing trends.
- Leverage sentiment analysis insights to highlight areas where customer satisfaction can be improved.
- Use the discount predictions to suggest how pricing strategies can be optimized over the next 5 days.
- Recommend promotional campaigns or marketing strategies that align with customer sentiments and competitive trends.
- Ensure the strategies are actionable, realistic, and geared toward increasing customer satisfaction, driving sales, and outperforming competitors.

Provide your recommendations in a structured format:
1. **Pricing Strategy**
2. **Promotional Campaign Ideas**
3. **Customer Satisfaction Recommendations**
    """

    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama3-8b-8192",
        "temperature": 0,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(data),
        headers=headers,
    )
    res = res.json()
    response = res["choices"][0]["message"]["content"]
    return response

#### Streamlit Dashboard ####
st.set_page_config(page_title="E-Commerce Competitor Strategy Dashboard", layout="wide")
st.title("E-Commerce Competitor Strategy Dashboard")
st.sidebar.header("Select a Product")

price_data = load_price_data()
reviews_data = load_reviews_data()

products = price_data["Title"].unique()
selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)

product_data = price_data[price_data["Title"] == selected_product]
product_reviews = reviews_data[reviews_data["Title"] == selected_product]

st.header(f"Competitor Analysis for {selected_product}")
st.subheader("Competitor Data")
st.table(product_data.tail(5))

if not product_reviews.empty:
    reviews = product_reviews["Review"].apply(lambda x: truncate_text(x, 512)).tolist()
    sentiments = analyze_sentiment(reviews)
    st.subheader("Customer Sentiment Analysis")
    sentiment_df = pd.DataFrame(sentiments)
    fig = px.bar(sentiment_df, x="label", title="Sentiment Analysis Results")
    st.plotly_chart(fig)
else:
    st.write("No reviews available for this product.")

# Preprocessing for forecasting
product_data.set_index("Date", inplace=True)
product_data_with_predictions = forecast_discounts_arima(product_data)

st.subheader("Competitor Current and Predicted Discounts")
st.table(product_data_with_predictions.tail(10))

recommendations = generate_strategy_recommendation(
    selected_product,
    product_data_with_predictions,
    sentiments if not product_reviews.empty else "No reviews available",
)

st.subheader("Strategic Recommendations")
st.write(recommendations)
send_to_slack(recommendations)