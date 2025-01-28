import os
import time
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import requests

# Helper functions
def fetch_soup(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return BeautifulSoup(response.content, "html.parser")
    except requests.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def get_title(soup):
    try:
        return soup.find("span", {"id": "productTitle"}).get_text(strip=True)
    except AttributeError:
        return "N/A"

def get_price(soup):
    try:
        price = soup.find("span", {"class": "a-price-whole"}).get_text(strip=True)
        return price
    except AttributeError:
        return "N/A"

def calculate_mrp(price):
    try:
        numeric_price = ''.join([c for c in price if c.isdigit()])
        return round(float(numeric_price) * 1.2, 2) if numeric_price else None
    except ValueError:
        return None

def get_rating(soup):
    try:
        return soup.find("span", {"class": "a-icon-alt"}).get_text(strip=True)
    except AttributeError:
        return "N/A"

def get_review_count(soup):
    try:
        return soup.find("span", {"id": "acrCustomerReviewText"}).get_text(strip=True)
    except AttributeError:
        return "N/A"

def get_reviews(soup, max_reviews=5):
    try:
        reviews = soup.find_all("span", {"data-hook": "review-body"})
        return [review.get_text(strip=True) for review in reviews[:max_reviews]]
    except AttributeError:
        return ["No reviews available"]

def scrape_amazon_product(url, collect_reviews=True):
    soup = fetch_soup(url)
    if not soup:
        return None

    price = get_price(soup)
    product_data = {
        "Title": get_title(soup),
        "Price": price,
        "MRP Price": calculate_mrp(price),
        "Rating": get_rating(soup),
        "Review Count": get_review_count(soup),
        "Reviews": get_reviews(soup) if collect_reviews else None,
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return product_data

def scrape_multiple_products_with_intervals(urls, output_file="amazon_products.csv", intervals=5, interval_seconds=180):
    review_data_collected = False

    for i in range(intervals):
        print(f"Scraping iteration {i + 1} of {intervals}...")
        data = []
        for url in urls:
            product_data = scrape_amazon_product(url, collect_reviews=not review_data_collected)
            if product_data:
                data.append(product_data)

        if data:
            # Save data to CSV
            df = pd.DataFrame(data)
            file_exists = os.path.exists(output_file)
            df.to_csv(output_file, mode='a', header=not file_exists, index=False)
            print(f"Data saved to {output_file}")

            # Save separate reviews if it's the first iteration
            if not review_data_collected:
                review_df = df[["Title", "Reviews"]]
                review_df.to_csv("review_data.csv", index=False)
                review_data_collected = True
                print("Reviews saved to review_data.csv")
        else:
            print("No data scraped.")

        # Wait before the next iteration
        if i < intervals - 1:
            time.sleep(interval_seconds)

# Test URLs
amazon_urls = [
    "https://www.amazon.in/boAt-Rockerz-480-Bluetooth-Headphones/dp/B0DGTSRX3R",
    "https://www.amazon.in/HP-i5-13420H-15-6-inch-Backlit-fa1319TX/dp/B0D1YJR2ZY",
    "https://www.amazon.in/Samsung-Galaxy-Ultra-Titanium-Storage/dp/B0DSKMKJV5",
]

# Scrape products and save data
scrape_multiple_products_with_intervals(amazon_urls, output_file="amazon_products.csv", intervals=5, interval_seconds=180)
