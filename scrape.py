import os
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import requests

# User-Agent for the HTTP request
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36"
)

# Helper function to fetch and parse HTML content from a given URL
def fetch_soup(url):
    headers = {
        "User-Agent": USER_AGENT,  # Use the provided user agent to avoid blocking
        "Accept-Language": "en-US,en;q=0.9"  # Request content in English
    }
    try:
        response = requests.get(url, headers=headers)  # Make a GET request to fetch webpage content
        response.raise_for_status()  # Raise an error for unsuccessful status codes
        return BeautifulSoup(response.content, "html.parser")  # Parse HTML content
    except requests.RequestException as e:
        print(f"Error fetching URL {url}: {e}")  # Print error message if request fails
        return None

# Extract product title from the webpage
def get_title(soup):
    try:
        return soup.find("span", {"id": "productTitle"}).get_text(strip=True)
    except AttributeError:
        return "N/A"  # Return "N/A" if title is not found

# Extract product price from the webpage
def get_price(soup):
    try:
        price = soup.find("span", {"class": "a-price-whole"}).get_text(strip=True)
        return price
    except AttributeError:
        return "N/A"

# Calculate the MRP price assuming a 20% markup
def calculate_mrp(price):
    try:
        numeric_price = ''.join([c for c in price if c.isdigit()])  # Remove non-numeric characters
        return round(float(numeric_price) * 1.2, 2) if numeric_price else None  # Apply 20% markup
    except ValueError:
        return None

# Extract product rating from the webpage
def get_rating(soup):
    try:
        return soup.find("span", {"class": "a-icon-alt"}).get_text(strip=True)
    except AttributeError:
        return "N/A"

# Extract the number of reviews from the webpage
def get_review_count(soup):
    try:
        return soup.find("span", {"id": "acrCustomerReviewText"}).get_text(strip=True)
    except AttributeError:
        return "N/A"

# Extract availability status from the webpage
def get_availability(soup):
    try:
        availability = soup.find("div", {"id": "availability"}).get_text(strip=True)
        return availability if availability else "In Stock"
    except AttributeError:
        return "N/A"

# Extract one user review from the product page
def get_one_review(soup):
    try:
        review_section = soup.find("span", {"data-hook": "review-body"})
        review_text = review_section.get_text(strip=True)
        return review_text
    except AttributeError:
        return "No reviews available"

# Main function to scrape product details
def scrape_amazon_product(url):
    soup = fetch_soup(url)  # Fetch webpage content
    if not soup:
        return None  # Return None if request failed

    price = get_price(soup)  # Extract price
    product_data = {
        "Title": get_title(soup),  # Extract product title
        "Price": price,  # Extract price
        "MRP Price": calculate_mrp(price),  # Calculate MRP
        "Rating": get_rating(soup),  # Extract rating
        "Review Count": get_review_count(soup),  # Extract number of reviews
        "Availability": get_availability(soup),  # Extract availability status
        "One Review": get_one_review(soup),  # Extract one user review
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Store current timestamp
    }
    return product_data

# Function to scrape multiple product URLs and save data to CSV
def scrape_multiple_products(urls, output_file="amazon_products.csv"):
    data = []  # Initialize empty list to store product data
    for url in urls:
        product_data = scrape_amazon_product(url)  # Scrape each product
        if product_data:
            data.append(product_data)  # Append scraped data to list

    if data:
        df = pd.DataFrame(data)  # Convert list to pandas DataFrame
        file_exists = os.path.exists(output_file)  # Check if file exists
        df.to_csv(output_file, mode='a', header=not file_exists, index=False)  # Save to CSV file
        print(f"Data saved to {output_file}")

        # Split data into separate files
        split_amazon_csv(output_file)
    else:
        print("No data scraped.")

# Function to split Amazon product data into reviews and pricing
def split_amazon_csv(input_file="amazon_products.csv", review_file="amazon_reviews.csv", price_file="amazon_price.csv"):
    try:
        # Read the original CSV
        df = pd.read_csv(input_file)
        
        # Extract the reviews data
        reviews_df = df[["Title", "Rating", "One Review"]]
        reviews_df.to_csv(review_file, index=False)
        print(f"Reviews saved to {review_file}")
        
        # Extract the price data
        price_df = df[["Date", "Title", "Price", "MRP Price", "Availability"]].drop_duplicates()
        price_df.to_csv(price_file, index=False)
        print(f"Price details saved to {price_file}")
    
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# List of Amazon product URLs to scrape
amazon_urls = [
    "https://www.amazon.in/boAt-Rockerz-480-Bluetooth-Headphones/dp/B0DGTSRX3R/ref=sr_1_4?crid=20R36NCKY9S89&keywords=headphones",
    "https://www.amazon.in/HP-i5-13420H-15-6-inch-Backlit-fa1319TX/dp/B0D1YJR2ZY/ref=sr_1_1_sspa?keywords=laptop",
    "https://www.amazon.in/Samsung-Galaxy-Ultra-Titanium-Storage/dp/B0DSKMKJV5/ref=sr_1_1_sspa?keywords=mobile"
]

# Start scraping process
scrape_multiple_products(amazon_urls, output_file="amazon_products.csv")
