import pandas as pd
import os
import argparse

def extract_conversations(dataset_path: str, brand_name: str, output_path: str):
    """
    Extracts customer query and brand reply pairs for a specific brand.
    """
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Brand replies
    brand_replies = df[df['author_id'] == brand_name].copy()
    print(f"Found {len(brand_replies)} total replies from {brand_name}.")
    
    # We only want replies that are responding to an inbound customer tweet
    brand_replies = brand_replies[brand_replies['in_response_to_tweet_id'].notna()]
    
    # Get the original customer tweets
    customer_tweets = df[df['tweet_id'].isin(brand_replies['in_response_to_tweet_id'])].copy()
    
    # Merge them on tweet_id (customer) == in_response_to_tweet_id (brand)
    # We only take the first reply if there are multiple, to keep it simple (1 to 1 mapping)
    brand_replies_dedup = brand_replies.drop_duplicates(subset=['in_response_to_tweet_id'])
    
    merged = pd.merge(
        customer_tweets[['tweet_id', 'author_id', 'text', 'created_at']],
        brand_replies_dedup[['in_response_to_tweet_id', 'text', 'created_at']],
        left_on='tweet_id',
        right_on='in_response_to_tweet_id',
        suffixes=('_customer', '_brand')
    )
    
    # Rename columns for clarity
    merged = merged.rename(columns={
        'text_customer': 'customer_query',
        'text_brand': 'brand_reply',
        'author_id': 'customer_id'
    })
    
    # Drop redundant column
    merged = merged.drop(columns=['in_response_to_tweet_id'])
    
    # Clean text (remove URLs, multiple spaces, etc. - simple cleaning)
    merged['customer_query'] = merged['customer_query'].str.replace(r'http\S+', '', regex=True).str.strip()
    merged['brand_reply'] = merged['brand_reply'].str.replace(r'http\S+', '', regex=True).str.strip()
    
    # Filter out empty strings after cleaning
    merged = merged[(merged['customer_query'] != '') & (merged['brand_reply'] != '')]
    
    print(f"Extracted {len(merged)} valid conversation pairs.")
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return merged

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess TWCS data.')
    parser.add_argument('--input', type=str, default='dataset/sample.csv', help='Path to input CSV')
    parser.add_argument('--brand', type=str, default='AppleSupport', help='Brand author_id to filter')
    parser.add_argument('--output', type=str, default='data/processed_conversations.csv', help='Path to output CSV')
    
    args = parser.parse_args()
    
    # If the large file exists, use it by default if user didn't specify
    input_path = args.input
    if input_path == 'dataset/sample.csv' and os.path.exists('dataset/twcs/twcs.csv'):
        input_path = 'dataset/twcs/twcs.csv'
        
    extract_conversations(input_path, args.brand, args.output)
