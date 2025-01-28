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
