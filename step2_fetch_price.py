import json
import time
from datetime import datetime, timedelta
import os
import asyncio
from playwright.async_api import async_playwright

def load_airdrop_data(file_path):
    """Load airdrop data from JSON file"""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data[:2]  # Take only first 2 results
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        return []

def timestamp_to_unix(timestamp_ms):
    """Convert millisecond timestamp to Unix timestamp (seconds)"""
    return timestamp_ms // 1000

async def fetch_price_history(page, coin_id, from_timestamp, to_timestamp):
    """
    Fetch price history using Playwright to avoid CORS issues
    
    Args:
        page: Playwright page instance
        coin_id: CoinGecko coin ID
        from_timestamp: Start timestamp in Unix seconds
        to_timestamp: End timestamp in Unix seconds
    """
    url = f"https://www.coingecko.com/price_charts/{coin_id}/usd/custom.json?from={from_timestamp}&to={to_timestamp}"
    
    try:
        print(f"Fetching data for {coin_id}...")
        print(f"URL: {url}")
        
        # Navigate to the API URL
        response = await page.goto(url, wait_until="domcontentloaded")
        
        if response.status != 200:
            print(f"Error: HTTP {response.status} for {coin_id}")
            return None
        
        # Get the JSON content from the page
        content = await page.content()
        
        # Extract JSON from the pre tag (CoinGecko returns JSON in <pre> tags)
        json_content = await page.evaluate("""
            () => {
                const preElement = document.querySelector('pre');
                return preElement ? preElement.textContent : document.body.textContent;
            }
        """)
        
        # Parse the JSON
        data = json.loads(json_content)
        print(f"Successfully fetched {len(data.get('stats', []))} data points for {coin_id}")
        return data
        
    except Exception as e:
        print(f"Error fetching data for {coin_id}: {e}")
        return None

def calculate_max_days_ahead():
    """
    Calculate maximum days we can add from the airdrop start date
    CoinGecko typically allows data up to current date
    """
    start_timestamp = 1743465600000  # Given timestamp in milliseconds
    start_date = datetime.fromtimestamp(start_timestamp / 1000)
    current_date = datetime.now()
    
    # Calculate days between start date and current date
    max_days = (current_date - start_date).days
    
    # For safety, use a reasonable limit (e.g., 30-90 days for good data resolution)
    recommended_days = min(max_days, 90)  # Max 90 days or until current date
    
    print(f"Start date: {start_date.strftime('%Y-%m-%d')}")
    print(f"Current date: {current_date.strftime('%Y-%m-%d')}")
    print(f"Maximum possible days: {max_days}")
    print(f"Recommended days for good resolution: {recommended_days}")
    
    return recommended_days

async def main():
    # File path
    json_file = "./data_airdrop_normalized.json"
    
    # Load airdrop data (first 2 items only)
    airdrop_data = load_airdrop_data(json_file)
    
    if not airdrop_data:
        print("No data to process")
        return
    
    # Calculate maximum days we can add
    max_days = calculate_max_days_ahead()
    
    # Set date range
    from_timestamp_ms = 1743465600000  # Given start timestamp
    from_timestamp = timestamp_to_unix(from_timestamp_ms)
    
    # Add specified days (you can modify this value)
    days_to_add = max_days  # Using maximum calculated days
    to_timestamp = from_timestamp + (days_to_add * 24 * 60 * 60)  # Add days in seconds
    
    print(f"\nFetching data from {datetime.fromtimestamp(from_timestamp)} to {datetime.fromtimestamp(to_timestamp)}")
    print(f"Date range: {days_to_add} days\n")
    
    # Create output directory
    output_dir = "price_history_tokens"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Playwright
    async with async_playwright() as p:
        # Launch browser (headless by default)
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Set user agent to avoid detection
        await page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Fetch data for each token
        successful_fetches = 0
        
        for token in airdrop_data:
            coin_id = token['id']
            symbol = token['symbol']
            name = token['name']
            
            print(f"\nProcessing: {name} ({symbol}) - ID: {coin_id}")
            
            # Fetch price history
            price_data = await fetch_price_history(page, coin_id, from_timestamp, to_timestamp)
            
            if price_data:
                # Save individual file for each token
                output_file = os.path.join(output_dir, f"{coin_id}.json")
                with open(output_file, 'w') as f:
                    json.dump(price_data, f, indent=2)
                
                print(f"Data saved to: {output_file}")
                print(f"Data points: {len(price_data.get('stats', []))}")
                successful_fetches += 1
            
            # Add delay to be respectful
            await asyncio.sleep(2)
        
        # Close browser
        await browser.close()
    
    print(f"\n=== SUMMARY ===")
    print(f"Processed {successful_fetches} tokens successfully out of {len(airdrop_data)} total")
    print(f"Data range: {days_to_add} days")
    print(f"Individual token files saved to: {output_dir}/")

if __name__ == "__main__":
    asyncio.run(main())