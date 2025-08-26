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

class BinancePriceFetcher:
    def __init__(self, rate_limit: float = 1.2):
        """
        Initialize the Binance price fetcher
        
        Args:
            rate_limit: Seconds to wait between API calls to respect rate limits
        """
        # Initialize both spot and futures exchanges
        self.spot_exchange = ccxt.binance({
            'rateLimit': int(rate_limit * 1000),
            'enableRateLimit': True,
        })
        
        self.futures_exchange = ccxt.binance({
            'rateLimit': int(rate_limit * 1000),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Use futures market
            }
        })
        
        self.rate_limit = rate_limit
        
    def load_airdrops(self, file_path: str) -> List[Dict]:
        """Load airdrops data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                airdrops = json.load(f)
            logger.info(f"Loaded {len(airdrops)} airdrops from {file_path}")
            return airdrops
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            return []
    
    def timestamp_to_datetime(self, timestamp: int) -> datetime:
        """Convert timestamp in milliseconds to datetime"""
        return datetime.fromtimestamp(timestamp / 1000)
    
    def is_pair_still_active(self, symbol: str, market_type: str = 'futures') -> bool:
        """
        Check if a trading pair is still active
        
        Args:
            symbol: Trading pair symbol
            market_type: 'spot' or 'futures'
            
        Returns:
            True if pair is still active, False otherwise
        """
        try:
            exchange = self.futures_exchange if market_type == 'futures' else self.spot_exchange
            markets = exchange.load_markets()
            
            if symbol not in markets:
                return False
                
            return markets[symbol].get('active', False)
            
        except Exception as e:
            logger.warning(f"Error checking if {symbol} is active: {e}")
            return False
    
    def fetch_futures_ohlcv_data(self, symbol: str, start_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from futures market until current date or until pair is delisted
        
        Args:
            symbol: Futures symbol (e.g., 'IOSTUSDT')
            start_date: Start date for fetching data
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Check if pair is still active
            is_active = self.is_pair_still_active(symbol, 'futures')
            end_date = datetime.now()
            
            logger.info(f"Fetching futures {symbol} from {start_date.date()} to {end_date.date()} (Active: {is_active})")
            
            # Convert to milliseconds timestamp
            since = int(start_date.timestamp() * 1000)
            
            all_ohlcv_data = []
            
            # Fetch data in chunks to handle large date ranges
            current_date = start_date
            chunk_days = 1000  # Fetch 1000 days at a time to avoid API limits
            
            while current_date < end_date:
                chunk_end = min(current_date + timedelta(days=chunk_days), end_date)
                chunk_since = int(current_date.timestamp() * 1000)
                
                try:
                    # Fetch OHLCV data from futures market with 1-day timeframe
                    ohlcv_data = self.futures_exchange.fetch_ohlcv(
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
                        logger.warning(f"Symbol {symbol} not found in futures market")
                        break
                    else:
                        logger.error(f"Error fetching chunk for {symbol}: {e}")
                        break
            
            if not all_ohlcv_data:
                logger.warning(f"No futures data found for {symbol}")
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
            
            logger.info(f"Fetched {len(df)} days of futures data for {symbol} (from {df['date'].min()} to {df['date'].max()})")
            return df
            
        except ccxt.BaseError as e:
            logger.error(f"CCXT error fetching futures {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching futures {symbol}: {e}")
            return None
    
    def fetch_spot_ohlcv_data(self, symbol: str, start_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from spot market until current date or until pair is delisted
        
        Args:
            symbol: Trading pair symbol (e.g., 'IOST/USDT')
            start_date: Start date for fetching data
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Check if pair is still active
            is_active = self.is_pair_still_active(symbol, 'spot')
            end_date = datetime.now()
            
            logger.info(f"Fetching spot {symbol} from {start_date.date()} to {end_date.date()} (Active: {is_active})")
            
            all_ohlcv_data = []
            
            # Fetch data in chunks to handle large date ranges
            current_date = start_date
            chunk_days = 1000  # Fetch 1000 days at a time to avoid API limits
            
            while current_date < end_date:
                chunk_end = min(current_date + timedelta(days=chunk_days), end_date)
                chunk_since = int(current_date.timestamp() * 1000)
                
                try:
                    # Fetch OHLCV data with 1-day timeframe
                    ohlcv_data = self.spot_exchange.fetch_ohlcv(
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
                        logger.warning(f"Symbol {symbol} not found in spot market")
                        break
                    else:
                        logger.error(f"Error fetching chunk for {symbol}: {e}")
                        break
            
            if not all_ohlcv_data:
                logger.warning(f"No spot data found for {symbol}")
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
            
            logger.info(f"Fetched {len(df)} days of spot data for {symbol} (from {df['date'].min()} to {df['date'].max()})")
            return df
            
        except ccxt.BaseError as e:
            logger.error(f"CCXT error fetching spot {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching spot {symbol}: {e}")
            return None
    
    def find_trading_pairs(self, asset: str) -> List[str]:
        """
        Find available trading pairs for an asset in spot market
        Prioritizes USDT, BUSD, BTC pairs
        """
        try:
            markets = self.spot_exchange.load_markets()
            pairs = []
            
            # Priority order for quote currencies
            quote_priorities = ['USDT', 'BUSD', 'BTC', 'ETH', 'BNB']
            
            for quote in quote_priorities:
                symbol = f"{asset}/{quote}"
                if symbol in markets and markets[symbol]['active']:
                    pairs.append(symbol)
            
            # Add any other available pairs
            for symbol in markets:
                if (symbol.startswith(f"{asset}/") and 
                    symbol not in pairs and 
                    markets[symbol]['active']):
                    pairs.append(symbol)
            
            return pairs
            
        except Exception as e:
            logger.error(f"Error finding trading pairs for {asset}: {e}")
            return []
    
    def convert_futures_symbol_to_spot(self, futures_symbol: str) -> str:
        """
        Convert futures symbol to spot symbol format
        E.g., 'IOSTUSDT' -> 'IOST/USDT'
        """
        # Most futures symbols end with USDT, BTC, ETH, etc.
        quote_assets = ['USDT', 'BUSD', 'BTC', 'ETH', 'BNB']
        
        for quote in quote_assets:
            if futures_symbol.endswith(quote):
                base = futures_symbol[:-len(quote)]
                return f"{base}/{quote}"
        
        # If no match found, assume USDT
        if futures_symbol.endswith('T'):  # Likely USDT
            base = futures_symbol[:-4]  # Remove 'USDT'
            return f"{base}/USDT"
        
        # Fallback
        return f"{futures_symbol}/USDT"
    
    def save_price_history_immediately(self, airdrop_id: int, asset: str, trading_pair: str, 
                                      start_date: datetime, price_data: pd.DataFrame, 
                                      market_type: str = 'spot',
                                      output_dir: str = './outputs/prices_history') -> str:
        """Save price history immediately after fetching"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare price history data
        price_history = {
            'airdrop_id': airdrop_id,
            'asset': asset,
            'trading_pair': trading_pair,
            'market_type': market_type,  # 'spot' or 'futures'
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
        
        # Save as JSON file
        json_file = os.path.join(output_dir, f'{asset}.json')
        with open(json_file, 'w') as f:
            json.dump(price_history, f, indent=2)
        
        date_range_str = f"from {price_history['date_range']['first_date']} to {price_history['date_range']['last_date']}"
        logger.info(f"✓ Price history saved to {json_file} ({date_range_str})")
        return json_file

    def process_airdrop(self, airdrop: Dict) -> Dict:
        """Process a single airdrop and fetch futures price data until current date"""
        airdrop_id = airdrop.get('airdropId')
        asset = airdrop.get('airdropAsset')
        start_timestamp = airdrop.get('airdropPeriodStart')
        futures_symbol = airdrop.get('futuresSymbol')
        
        if not all([airdrop_id, asset, start_timestamp]):
            logger.warning(f"Missing required data for airdrop: {airdrop}")
            return {'airdrop_id': airdrop_id, 'asset': asset, 'status': 'error', 'message': 'Missing required data'}
        
        if not futures_symbol:
            logger.warning(f"No futures symbol found for {asset}")
            return {'airdrop_id': airdrop_id, 'asset': asset, 'status': 'no_futures', 'message': 'No futures symbol available'}
        
        # Convert timestamp to datetime
        start_date = self.timestamp_to_datetime(start_timestamp)
        
        logger.info(f"Processing Airdrop ID {airdrop_id}: {asset} (futures only, fetching until current date)")
        
        # Fetch futures data until current date
        logger.info(f"Fetching futures data for {futures_symbol}")
        try:
            futures_data = self.fetch_futures_ohlcv_data(futures_symbol, start_date)
            
            if futures_data is not None and not futures_data.empty:
                # Save futures price history immediately
                saved_file = self.save_price_history_immediately(
                    airdrop_id, asset, futures_symbol, start_date, futures_data, 'futures'
                )
                
                return {
                    'airdrop_id': airdrop_id,
                    'asset': asset,
                    'trading_pair': futures_symbol,
                    'market_type': 'futures',
                    'status': 'success',
                    'data_points': len(futures_data),
                    'start_date': start_date.isoformat(),
                    'end_date': futures_data['datetime'].max().isoformat(),
                    'date_range': f"{futures_data['date'].min()} to {futures_data['date'].max()}",
                    'saved_file': saved_file,
                    'message': f'Successfully fetched {len(futures_data)} days of futures data until {futures_data["date"].max()}'
                }
            else:
                logger.warning(f"No futures data found for {futures_symbol}")
                return {'airdrop_id': airdrop_id, 'asset': asset, 'status': 'no_data', 'message': f'No futures data found for {futures_symbol}'}
                
        except Exception as e:
            logger.error(f"Failed to fetch futures data for {futures_symbol}: {e}")
            return {'airdrop_id': airdrop_id, 'asset': asset, 'status': 'failed', 'message': f'Failed to fetch futures data: {str(e)}'}
    
    def save_results(self, results: List[Dict], output_dir: str = './outputs'):
        """Process results without saving summary file"""
        # Just log the completion, no file saving
        successful = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"Price fetching completed: {successful}/{len(results)} successful")
        
        # Log date ranges for successful fetches
        for result in results:
            if result['status'] == 'success':
                logger.info(f"✓ {result['asset']}: {result.get('date_range', 'N/A')}")
    
    def run(self, input_file: str = './outputs/airdrops_list.json', output_dir: str = './outputs'):
        """Main execution function"""
        logger.info("Starting Binance futures price data fetching (until current date)...")
        
        # Load airdrops
        airdrops = self.load_airdrops(input_file)
        if not airdrops:
            logger.error("No airdrops to process")
            return
        
        results = []
        
        # Process each airdrop
        for i, airdrop in enumerate(airdrops, 1):
            logger.info(f"Processing airdrop {i}/{len(airdrops)}")
            result = self.process_airdrop(airdrop)
            results.append(result)
            
            # Rate limiting between airdrops
            if i < len(airdrops):
                time.sleep(self.rate_limit)
        
        # Process results (no file saving)
        self.save_results(results, output_dir)
        
        # Print summary
        successful = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"Completed! Successfully fetched futures data for {successful}/{len(results)} airdrops")
        
        return results


def main():
    """Main function"""
    # Configuration
    INPUT_FILE = './outputs/airdrops_list.json'
    OUTPUT_DIR = './outputs'
    RATE_LIMIT = 1.2  # Seconds between API calls
    
    # Initialize fetcher
    fetcher = BinancePriceFetcher(rate_limit=RATE_LIMIT)
    
    try:
        # Run the fetcher
        results = fetcher.run(INPUT_FILE, OUTPUT_DIR)
        
        # Print final summary
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        
        status_counts = {}
        successful = 0
        
        for result in results:
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == 'success':
                successful += 1
        
        print("STATUS SUMMARY:")
        for status, count in status_counts.items():
            print(f"  {status.upper()}: {count}")
        
        print("\nMARKET TYPE SUMMARY:")
        print(f"  FUTURES: {successful}")
        
        no_futures = sum(1 for r in results if r['status'] == 'no_futures')
        if no_futures > 0:
            print(f"  NO FUTURES SYMBOL: {no_futures}")
        
        # Print date ranges for successful fetches
        print("\nDATE RANGES FOR SUCCESSFUL FETCHES:")
        for result in results:
            if result['status'] == 'success':
                print(f"  {result['asset']}: {result.get('date_range', 'N/A')} ({result.get('data_points', 0)} days)")
        
        print("="*50)
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()