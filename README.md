# E-Commerce Competitor Strategy Dashboard

This project is designed to help e-commerce businesses analyze competitor strategies, sentiment analysis, and predict future discounts for products. By combining various models such as Random Forest and ARIMA, the application can assist businesses in making data-driven decisions for optimizing their pricing, promotions, and customer satisfaction.

## Table of Contents

- [Introduction](#introduction)
- [Setup Instructions](#setup-instructions)
- [How to Use](#how-to-use)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [License](#license)

## Introduction

The **E-Commerce Competitor Strategy Dashboard** provides businesses with actionable insights by analyzing competitor data, customer sentiment, and discount forecasting. It leverages machine learning models for predictive analysis and sentiment analysis models to generate insights and recommend strategies that align with market trends.

---

### ![Dashboard Image](![Streamlit1](https://github.com/user-attachments/assets/c987926f-b697-4d7d-82a4-c394bcc2f6e7))

---

## Setup Instructions

1. **Clone the repository** to your local machine:
    ```bash
    git clone https://github.com/yourusername/e-commerce-competitor-strategy-dashboard.git
    cd e-commerce-competitor-strategy-dashboard
    ```

2. **Install dependencies** using pip:
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure API keys**:
    - Create an `.env` file and add your Groq API Key and Slack Webhook URL.
    ```plaintext
    API_KEY=your_groq_api_key
    SLACK_WEBHOOK=your_slack_webhook_url
    ```

4. **Run the app**:
    ```bash
    streamlit run app.py
    ```

---

## How to Use

1. Launch the app using Streamlit.
2. Select a product from the sidebar.
3. View the competitor data and sentiment analysis results.
4. See the predicted future discounts for the selected product.
5. Generate strategic recommendations to optimize pricing, promotions, and customer satisfaction.

---

### ![Product Analysis Example]([Streamlit2](https://github.com/user-attachments/assets/490a796e-8775-45fa-b0a7-a01959638a53)))

---

## Features

- **Competitor Data Analysis**: Load and display competitor pricing and discount strategies.
- **Sentiment Analysis**: Use NLP models to analyze customer reviews and extract sentiment.
- **Forecasting**: Predict future discount trends using ARIMA.
- **Strategy Recommendations**: Get actionable strategies based on competitor analysis and sentiment data.
- **Amazon CSV Splitter**: A script to split the original Amazon product CSV into two separate files for reviews and pricing data.

---

### ![Sentiment Analysis Example](![Streamlit3](https://github.com/user-attachments/assets/c8bd3bb4-3b27-4e6a-ac54-048d912d1d8d))

---

## Technologies Used

- **Python**: The primary programming language.
- **Streamlit**: For building the interactive web dashboard.
- **Plotly**: For visualizing sentiment analysis results.
- **Scikit-Learn**: For training predictive models using Random Forest Regressor.
- **ARIMA (Statsmodels)**: For time series forecasting of discounts.
- **Transformers (Hugging Face)**: For sentiment analysis using BERT-based models.

---

## Amazon CSV Splitter

You can use the following script to split an Amazon product CSV into two separate files: one for **reviews** and one for **pricing details**.

### `split_amazon_csv` function:

```python
import pandas as pd

def split_amazon_csv(input_file="amazon_products.csv", review_file="amazon_reviews.csv", price_file="amazon_price.csv"):
    try:
        # Read the original CSV
        df = pd.read_csv(input_file)
        
        # Extract the reviews data
        reviews_df = df[["Title", "Rating", "Review"]]
        reviews_df.to_csv(review_file, index=False)
        print(f"Reviews saved to {review_file}")
        
        # Extract the price data
        price_df = df[["Date", "Title", "Price", "MRP Price", "Discount (%)", "Availability"]].drop_duplicates()
        price_df.to_csv(price_file, index=False)
        print(f"Price details saved to {price_file}")
    
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the function to split the CSV
split_amazon_csv(input_file="amazon_products.csv", review_file="amazon_reviews.csv", price_file="amazon_price.csv")

