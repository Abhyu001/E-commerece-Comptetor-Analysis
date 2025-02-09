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
API_KEY = "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"  # Groq API Key
SLACK_WEBHOOK = "https://hooks.slack.com/services/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"


# Function to truncate the length of text to the specified length
def truncate_text(text, max_length=512):
    return text[:max_length]


# Loading the competitor data from a CSV file
def load_competitor_data():
    """Load competitor data from a CSV file."""
    data = pd.read_csv("amazon_products.csv")  # Loading the competitor product data
    print(data.head())  # Print the first few rows of the data for verification
    return data


# Loading the reviews data from a CSV file
def load_reviews_data():
    """Load reviews data from a CSV file."""
    reviews = pd.read_csv("review_data.csv")  # Loading the reviews data
    return reviews


# Applying sentiment analysis pipeline using a specific model
def analyze_sentiment(reviews):
    """Analyze customer sentiment for reviews."""
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0f"
    )
    return sentiment_pipeline([str(review) for review in reviews])


# Train the model using Random Forest Regressor to predict competitor pricing strategy
def train_predictive_model(data):
    """Train a predictive model for competitor pricing strategy."""
    data["Discount"] = data["Discount"].str.replace("%", "").astype(float)  # Remove % and convert to float
    data["MRP Price"] = data["MRP Price"].astype(int)  # Convert MRP Price to integer
    data["Predicted_Discount"] = data["Discount"] + (data["MRP Price"] * 0.05).round(2)  # Predicted Discount

    X = data[["MRP Price", "Discount"]]  # Features for training
    y = data["Predicted_Discount"]  # Target variable
    print(X)  # Print the features data for verification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, train_size=0.8
    )

    model = RandomForestRegressor(random_state=42)  # Create a RandomForestRegressor model
    model.fit(X_train, y_train)  # Train the model with the training data
    return model


# Fitting ARIMA model for forecasting discounts
def forecast_discounts_arima(data, future_days=5):
    """
    Forecast future discounts using ARIMA.
    :param data: DataFrame containing historical discount data (with a datetime index).
    :param future_days: Number of days to forecast.
    :return: DataFrame with historical and forecasted discounts.
    """
    data = data.sort_index()  # Sort the data by index (date)

    # Convert the discount column to numeric, handling errors gracefully
    data["Discount"] = pd.to_numeric(data["Discount"], errors="coerce")
    data = data.dropna(subset=["Discount"])  # Drop rows with missing discount values

    discount_series = data["Discount"]  # Extract discount series for forecasting
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)  # Convert index to datetime if it's not already
        except Exception as e:
            raise ValueError("Index must be datetime or convertible to datetime.") from e

    # ARIMA(p,d,q)
    model = ARIMA(discount_series, order=(5, 1, 0))  # Define ARIMA model parameters
    model_fit = model.fit()  # Fit the ARIMA model

    forecast = model_fit.forecast(steps=future_days)  # Forecast future discounts
    future_dates = pd.date_range(
        start=discount_series.index[-1] + pd.Timedelta(days=1), periods=future_days
    )  # Generate future dates for the forecast

    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast})  # Prepare forecast dataframe
    forecast_df.set_index("Date", inplace=True)  # Set Date as the index for forecast dataframe

    return forecast_df


# Webhook contains its message payload and its head
def send_to_slack(data):
    """Send a message to Slack."""
    payload = {"text": data}  # Prepare the message payload
    response = requests.post(
        SLACK_WEBHOOK,
        data=json.dumps(payload),  # Send the message as JSON data
        headers={"Content-Type": "application/json"},
    )


# Generating strategic recommendations based on competitor data and sentiment analysis
def generate_strategy_recommendation(product_name, competitor_data, sentiment):
    """Generate strategic recommendations using an LLM."""
    date = datetime.now()  # Get the current date
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
        "model": "llama3-8b-8192",  # Define the model for generating recommendations
        "temperature": 0,  # Set the temperature to 0 for deterministic output
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",  # API endpoint for Groq service
        data=json.dumps(data),  # Send data as JSON
        headers=headers,  # Add headers
    )
    res = res.json()  # Parse the response as JSON
    response = res["choices"][0]["message"]["content"]  # Extract the recommendation content from the response
    return response


####--------------------------------------------------##########
# Streamlit app title 
st.set_page_config(page_title="E-Commerce Competitor Strategy Dashboard", layout="wide")

st.title("E-Commerce Competitor Strategy Dashboard")  # Title of the app
st.sidebar.header("Select a Product")  # Sidebar header for product selection

products = [
    "boAt Rockerz 480 w/RGB LEDs, 6 Light Modes, 40mm Drivers, Beast Mode, 60hrs Playback, ENx Tech, BT v5.3, Adaptive Fit & Easy Access Controls, Bluetooth Headphones(Black Sabre)",
    "HP Victus, 13th Gen Intel Core i5-13420H, 6GB NVIDIA RTX 4050, 16GB DDR4, 512GB SSD (Win11, Office 21, Silver, 2.29kg) 144Hz, 9MS, IPS, 15.6-inch(39.6cm) FHD Gaming Laptop, Enhanced Cooling, fa1319TX",
    "HAVAI Thunder 85 Desert Cooler - 75 Litres, 16 Inch Blade, Black",
    "Samsung Galaxy M05 (Mint Green, 4GB RAM, 64 GB Storage) | 50MP Dual Camera | Bigger 6.7 HD+ Display | 5000mAh Battery | 25W Fast Charging | 2 Gen OS Upgrade & 4 Year Security Update | Without Charger",
    "iQOO Z9x 5G (Tornado Green, 6GB RAM, 128GB Storage) | Snapdragon 6 Gen 1 with 560k+ AnTuTu Score | 6000mAh Battery with 7.99mm Slim Design | 44W FlashCharge"
]
selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)  # Select product from dropdown

# Calling functions part of Driver Code
competitor_data = load_competitor_data()  # Load competitor data
reviews_data = load_reviews_data()  # Load reviews data

# Filter data for the selected product
product_data = competitor_data[competitor_data["Title"] == selected_product]
product_reviews = reviews_data[reviews_data["Title"] == selected_product]

# Displaying competitor analysis
st.header(f"Competitor Analysis for {selected_product}")
st.subheader("Competitor Data")
st.table(product_data.tail(5))  # Show last 5 rows of competitor data

# Displaying customer sentiment analysis if reviews are available
if not product_reviews.empty:
    product_reviews["Reviews"] = product_reviews["Reviews"].apply(
        lambda x: truncate_text(x, 512)  # Truncate reviews to 512 characters
    )
    reviews = product_reviews["Reviews"].tolist()
    sentiments = analyze_sentiment(reviews)  # Perform sentiment analysis

    st.subheader("Customer Sentiment Analysis")
    sentiment_df = pd.DataFrame(sentiments)
    fig = px.bar(sentiment_df, x="label", title="Sentiment Analysis Results")  # Visualize sentiment
    st.plotly_chart(fig)  # Display the chart
else:
    st.write("No reviews available for this product.")  # If no reviews available

# Preprocessing the competitor data for forecasting
product_data["Date"] = pd.to_datetime(product_data["Date"], errors="coerce")
product_data = product_data.dropna(subset=["Date"])
product_data.set_index("Date", inplace=True)
product_data = product_data.sort_index()

product_data["Discount"] = pd.to_numeric(product_data["Discount"], errors="coerce")
product_data = product_data.dropna(subset=["Discount"])

# Forecasting Model to predict future discounts
product_data_with_predictions = forecast_discounts_arima(product_data)

# Displaying competitor current and predicted discounts
st.subheader("Competitor Current and Predicted Discounts")
st.table(product_data_with_predictions.tail(10))

# Generating and displaying strategic recommendations
recommendations = generate_strategy_recommendation(
    selected_product,
    product_data_with_predictions,
    sentiments if not product_reviews.empty else "No reviews available",
)
st.subheader("Strategic Recommendations")
st.write(recommendations)  # Show the recommendations

# Sending recommendations to Slack
send_to_slack(recommendations)
