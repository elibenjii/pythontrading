import ccxt
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoPriceFetcher:
    def __init__(self, rate_limit: float = 1.2):
        """
        Initialize the crypto price fetcher
        
        Args:
            rate_limit: Seconds to wait between API calls to respect rate limits
        """
        # Initialize Binance exchange
        self.exchange = ccxt.binance({
            'rateLimit': int(rate_limit * 1000),
            'enableRateLimit': True,
        })
        
        self.rate_limit = rate_limit
        
        # Define the cryptocurrencies we want to fetch
        self.cryptocurrencies = [
            {
                'name': 'Bitcoin',
                'symbol': 'BTC',
                'trading_pair': 'BTC/USDT'
            },
            {
                'name': 'Ethereum',
                'symbol': 'ETH',
                'trading_pair': 'ETH/USDT'
            }
        ]
    
    def timestamp_to_datetime(self, timestamp: int) -> datetime:
        """Convert timestamp in milliseconds to datetime"""
        return datetime.fromtimestamp(timestamp / 1000)
    
    def fetch_ohlcv_data(self, symbol: str, start_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from start_date until current date
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            start_date: Start date for fetching data
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            end_date = datetime.now()
            
            logger.info(f"Fetching {symbol} from {start_date.date()} to {end_date.date()}")
            
            all_ohlcv_data = []
            
            # Fetch data in chunks to handle large date ranges
            current_date = start_date
            chunk_days = 1000  # Fetch 1000 days at a time to avoid API limits
            
            while current_date < end_date:
                chunk_end = min(current_date + timedelta(days=chunk_days), end_date)
                chunk_since = int(current_date.timestamp() * 1000)
                
                try:
                    # Fetch OHLCV data with 1-day timeframe
                    ohlcv_data = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe='1d',
                        since=chunk_since,
                        limit=chunk_days
                    )
                    
                    if ohlcv_data:
                        # Filter data for this chunk's date range
                        chunk_until = int(chunk_end.timestamp() * 1000)
                        filtered_chunk = [candle for candle in ohlcv_data if candle[0] <= chunk_until]
                        all_ohlcv_data.extend(filtered_chunk)
                        
                        # If we got less data than expected, we might have reached the end
                        if len(ohlcv_data) < chunk_days:
                            break
                    else:
                        # No more data available
                        break
                        
                    # Move to next chunk
                    current_date = chunk_end
                    
                    # Small delay between chunks
                    time.sleep(0.1)
                    
                except ccxt.BaseError as e:
                    if "Invalid symbol" in str(e) or "symbol not found" in str(e).lower():
                        logger.warning(f"Symbol {symbol} not found")
                        break
                    else:
                        logger.error(f"Error fetching chunk for {symbol}: {e}")
                        break
            
            if not all_ohlcv_data:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Remove duplicates based on timestamp and sort
            seen_timestamps = set()
            unique_data = []
            for candle in all_ohlcv_data:
                if candle[0] not in seen_timestamps:
                    unique_data.append(candle)
                    seen_timestamps.add(candle[0])
            
            unique_data.sort(key=lambda x: x[0])  # Sort by timestamp
            
            # Convert to DataFrame
            df = pd.DataFrame(unique_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['date'] = df['datetime'].dt.date
            
            logger.info(f"Fetched {len(df)} days of data for {symbol} (from {df['date'].min()} to {df['date'].max()})")
            return df
            
        except ccxt.BaseError as e:
            logger.error(f"CCXT error fetching {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol}: {e}")
            return None
    
    def save_price_history(self, crypto_info: Dict, start_date: datetime, 
                          price_data: pd.DataFrame, output_dir: str = './outputs') -> str:
        """Save price history to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare price history data
        price_history = {
            'name': crypto_info['name'],
            'symbol': crypto_info['symbol'],
            'trading_pair': crypto_info['trading_pair'],
            'start_date': start_date.isoformat(),
            'end_date': price_data['datetime'].max().isoformat() if not price_data.empty else None,
            'data_points': len(price_data),
            'date_range': {
                'first_date': price_data['date'].min().isoformat() if not price_data.empty else None,
                'last_date': price_data['date'].max().isoformat() if not price_data.empty else None,
            },
            'price_data': []
        }
        
        # Convert DataFrame rows to list of dictionaries
        for _, row in price_data.iterrows():
            price_history['price_data'].append({
                'timestamp': int(row['timestamp']),
                'datetime': row['datetime'].isoformat(),
                'date': row['date'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        
        # Save as JSON file directly in outputs folder
        json_file = os.path.join(output_dir, f'{crypto_info["symbol"]}.json')
        with open(json_file, 'w') as f:
            json.dump(price_history, f, indent=2)
        
        date_range_str = f"from {price_history['date_range']['first_date']} to {price_history['date_range']['last_date']}"
        logger.info(f"✓ {crypto_info['name']} price history saved to {json_file} ({date_range_str})")
        return json_file

    def process_cryptocurrency(self, crypto_info: Dict, start_date: datetime) -> Dict:
        """Process a single cryptocurrency and fetch price data"""
        name = crypto_info['name']
        symbol = crypto_info['symbol']
        trading_pair = crypto_info['trading_pair']
        
        logger.info(f"Processing {name} ({symbol})")
        
        try:
            # Fetch price data
            price_data = self.fetch_ohlcv_data(trading_pair, start_date)
            
            if price_data is not None and not price_data.empty:
                # Save price history
                saved_file = self.save_price_history(crypto_info, start_date, price_data)
                
                return {
                    'name': name,
                    'symbol': symbol,
                    'trading_pair': trading_pair,
                    'status': 'success',
                    'data_points': len(price_data),
                    'start_date': start_date.isoformat(),
                    'end_date': price_data['datetime'].max().isoformat(),
                    'date_range': f"{price_data['date'].min()} to {price_data['date'].max()}",
                    'saved_file': saved_file,
                    'first_price': float(price_data.iloc[0]['close']),
                    'last_price': float(price_data.iloc[-1]['close']),
                    'price_change': float(price_data.iloc[-1]['close'] - price_data.iloc[0]['close']),
                    'price_change_percent': float((price_data.iloc[-1]['close'] - price_data.iloc[0]['close']) / price_data.iloc[0]['close'] * 100),
                    'highest_price': float(price_data['high'].max()),
                    'lowest_price': float(price_data['low'].min()),
                    'message': f'Successfully fetched {len(price_data)} days of data'
                }
            else:
                logger.warning(f"No data found for {trading_pair}")
                return {
                    'name': name,
                    'symbol': symbol,
                    'trading_pair': trading_pair,
                    'status': 'no_data',
                    'message': f'No data found for {trading_pair}'
                }
                
        except Exception as e:
            logger.error(f"Failed to fetch data for {trading_pair}: {e}")
            return {
                'name': name,
                'symbol': symbol,
                'trading_pair': trading_pair,
                'status': 'failed',
                'message': f'Failed to fetch data: {str(e)}'
            }
    
    def log_summary(self, results: List[Dict]):
        """Log summary of results without saving file"""
        # Log summary
        successful = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"Price fetching completed: {successful}/{len(results)} successful")
        
        # Log details for successful fetches
        for result in results:
            if result['status'] == 'success':
                logger.info(f"✓ {result['name']}: {result.get('date_range', 'N/A')} "
                           f"(${result['first_price']:,.2f} → ${result['last_price']:,.2f}, "
                           f"{result['price_change_percent']:+.2f}%)")
    
    def run(self, start_year: int = 2019, output_dir: str = './outputs'):
        """Main execution function"""
        logger.info(f"Starting Bitcoin and Ethereum price data fetching from {start_year}...")
        
        # Set start date to January 1st of the specified year
        start_date = datetime(start_year, 1, 1)
        
        results = []
        
        # Process each cryptocurrency
        for i, crypto_info in enumerate(self.cryptocurrencies, 1):
            logger.info(f"Processing cryptocurrency {i}/{len(self.cryptocurrencies)}")
            result = self.process_cryptocurrency(crypto_info, start_date)
            results.append(result)
            
            # Rate limiting between cryptocurrencies
            if i < len(self.cryptocurrencies):
                time.sleep(self.rate_limit)
        
        # Log summary
        self.log_summary(results)
        
        # Print final summary
        successful = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"Completed! Successfully fetched data for {successful}/{len(results)} cryptocurrencies")
        
        return results


def main():
    """Main function"""
    # Configuration
    START_YEAR = 2019  # Fetch data from this year
    OUTPUT_DIR = './outputs'
    RATE_LIMIT = 1.2  # Seconds between API calls
    
    # Initialize fetcher
    fetcher = CryptoPriceFetcher(rate_limit=RATE_LIMIT)
    
    try:
        # Run the fetcher
        results = fetcher.run(START_YEAR, OUTPUT_DIR)
        
        # Print final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY - BITCOIN & ETHEREUM PRICE DATA")
        print("="*70)
        
        print(f"DATA PERIOD: January 1, {START_YEAR} to {datetime.now().strftime('%B %d, %Y')}")
        print()
        
        successful_results = [r for r in results if r['status'] == 'success']
        
        if successful_results:
            print("PRICE SUMMARY:")
            for result in successful_results:
                print(f"\n{result['name'].upper()} ({result['symbol']}):")
                print(f"  Trading Pair: {result['trading_pair']}")
                print(f"  Data Points: {result['data_points']:,} days")
                print(f"  Date Range: {result['date_range']}")
                print(f"  First Price: ${result['first_price']:,.2f}")
                print(f"  Last Price: ${result['last_price']:,.2f}")
                print(f"  Total Change: ${result['price_change']:,.2f} ({result['price_change_percent']:+.2f}%)")
                print(f"  Highest Price: ${result['highest_price']:,.2f}")
                print(f"  Lowest Price: ${result['lowest_price']:,.2f}")
        
        # Status summary
        print(f"\nSTATUS SUMMARY:")
        status_counts = {}
        for result in results:
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        for status, count in status_counts.items():
            print(f"  {status.upper()}: {count}")
        
        print("\nFILES SAVED:")
        for result in successful_results:
            if 'saved_file' in result:
                print(f"  {result['symbol']}: {result['saved_file']}")
        
        print("="*70)
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()