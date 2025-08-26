import json
import requests
import time
from typing import List, Dict, Any, Optional

def load_airdrop_data_from_api() -> List[Dict[Any, Any]]:
    """Load airdrop data from Binance API."""
    url = "https://www.binance.com/bapi/asset/v1/friendly/asset-service/airdrop/list"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if the response has the expected structure
        if data.get('code') != '000000':
            print(f"API returned error code: {data.get('code')} - {data.get('message')}")
            return []
        
        airdrop_data = data.get('data', [])
        print(f"Successfully fetched {len(airdrop_data)} airdrop entries from API")
        return airdrop_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching airdrop data from API: {e}")
        return []
    except json.JSONDecodeError:
        print("Error: Invalid JSON response from Binance airdrop API")
        return []

def get_binance_futures_symbols() -> Dict[str, Dict[str, Any]]:
    """Fetch all available symbols from Binance Futures API with symbol mapping."""
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        symbols_map = {}
        
        # Extract base assets and map to their trading symbols
        for symbol_info in data.get('symbols', []):
            if symbol_info.get('status') == 'TRADING' and symbol_info.get('contractType') == 'PERPETUAL':
                base_asset = symbol_info.get('baseAsset')
                symbol = symbol_info.get('symbol')
                quote_asset = symbol_info.get('quoteAsset')
                
                if base_asset and symbol:
                    # Store symbol info for this base asset
                    if base_asset not in symbols_map:
                        symbols_map[base_asset] = []
                    
                    symbols_map[base_asset].append({
                        'symbol': symbol,
                        'baseAsset': base_asset,
                        'quoteAsset': quote_asset,
                        'pricePrecision': symbol_info.get('pricePrecision', 2),
                        'quantityPrecision': symbol_info.get('quantityPrecision', 3),
                        'maintMarginPercent': symbol_info.get('maintMarginPercent'),
                        'onboardDate': symbol_info.get('onboardDate')
                    })
        
        print(f"Found {len(symbols_map)} base assets with {sum(len(v) for v in symbols_map.values())} trading pairs on Binance Futures")
        return symbols_map
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Binance Futures data: {e}")
        return {}
    except json.JSONDecodeError:
        print("Error: Invalid JSON response from Binance API")
        return {}

def find_futures_symbol_for_asset(asset: str, symbols_map: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Find the futures symbol information for a given asset."""
    # Handle special cases for certain tokens
    asset_mappings = {
        '1000PEPPER': 'PEPPER',
        '1000SATS': 'SATS',
        # Add more mappings as needed
    }
    
    # Helper function to get the preferred trading pair (prioritize USDT)
    def get_preferred_symbol(symbol_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Prefer USDT pairs, then USDC, then others
        quote_priority = ['USDT', 'USDC', 'BUSD', 'USD']
        
        for quote in quote_priority:
            for symbol_info in symbol_list:
                if symbol_info['quoteAsset'] == quote:
                    return symbol_info
        
        # If no preferred quote asset found, return the first one
        return symbol_list[0] if symbol_list else None
    
    # Check original asset name
    if asset in symbols_map:
        return get_preferred_symbol(symbols_map[asset])
    
    # Check mapped asset name
    mapped_asset = asset_mappings.get(asset)
    if mapped_asset and mapped_asset in symbols_map:
        return get_preferred_symbol(symbols_map[mapped_asset])
    
    # Check if asset starts with numbers (like 1000PEPPER)
    if asset.startswith('1000'):
        base_asset = asset[4:]  # Remove '1000' prefix
        if base_asset in symbols_map:
            return get_preferred_symbol(symbols_map[base_asset])
    
    return None

def filter_airdrops_by_futures(airdrop_data: List[Dict[Any, Any]], symbols_map: Dict[str, List[Dict[str, Any]]]) -> List[Dict[Any, Any]]:
    """Filter airdrops to only include assets listed on Binance Futures and add futures symbol info."""
    filtered_airdrops = []
    
    for airdrop in airdrop_data:
        airdrop_asset = airdrop.get('airdropAsset', '')
        
        futures_symbol_info = find_futures_symbol_for_asset(airdrop_asset, symbols_map)
        
        if futures_symbol_info:
            # Add futures symbol information to the airdrop data
            enhanced_airdrop = airdrop.copy()
            enhanced_airdrop['futuresSymbol'] = futures_symbol_info['symbol']
            enhanced_airdrop['futuresQuoteAsset'] = futures_symbol_info['quoteAsset']
            enhanced_airdrop['futuresPricePrecision'] = futures_symbol_info['pricePrecision']
            enhanced_airdrop['futuresQuantityPrecision'] = futures_symbol_info['quantityPrecision']
            enhanced_airdrop['futuresMaintMarginPercent'] = futures_symbol_info['maintMarginPercent']
            enhanced_airdrop['futuresOnboardDate'] = futures_symbol_info['onboardDate']
            
            filtered_airdrops.append(enhanced_airdrop)
            print(f"✓ Including: {airdrop_asset} -> {futures_symbol_info['symbol']}")
        else:
            print(f"✗ Excluding: {airdrop_asset} (not on Binance Futures)")
    
    return filtered_airdrops

def save_filtered_data(filtered_data: List[Dict[Any, Any]], output_path: str) -> None:
    """Save filtered data to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(filtered_data, file, indent=2, ensure_ascii=False)
        print(f"\nFiltered data saved to: {output_path}")
        print(f"Entries saved: {len(filtered_data)} entries")
    except Exception as e:
        print(f"Error saving file: {e}")

def get_futures_price(symbol: str) -> Optional[float]:
    """Get current futures price for a symbol (helper function for later use)."""
    url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return float(data.get('price', 0))
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

def main():
    """Main function to execute the filtering process."""
    output_file = "./outputs/airdrops_list.json"
    
    print("Binance Futures Airdrop Filter")
    print("=" * 40)
    
    # Load airdrop data from API
    print("Fetching airdrop data from Binance API...")
    airdrop_data = load_airdrop_data_from_api()
    
    if not airdrop_data:
        print("No airdrop data loaded. Exiting.")
        return
    
    print(f"Loaded {len(airdrop_data)} airdrop entries from API")
    
    # Get Binance Futures symbols
    print("\nFetching Binance Futures trading symbols...")
    symbols_map = get_binance_futures_symbols()
    
    if not symbols_map:
        print("Failed to fetch Binance Futures symbols. Exiting.")
        return
    
    # Filter airdrops
    print("\nFiltering airdrops...")
    print("-" * 40)
    filtered_airdrops = filter_airdrops_by_futures(airdrop_data, symbols_map)
    
    # Save filtered data
    save_filtered_data(filtered_airdrops, output_file)
    
    # Summary
    print(f"\nSummary:")
    print(f"Total airdrops processed: {len(airdrop_data)}")
    print(f"Airdrops with Futures-listed assets: {len(filtered_airdrops)}")
    print(f"Filtered out: {len(airdrop_data) - len(filtered_airdrops)}")
    
    if filtered_airdrops:
        print(f"\nAssets with their futures symbols:")
        for airdrop in filtered_airdrops:
            asset = airdrop.get('airdropAsset', '')
            futures_symbol = airdrop.get('futuresSymbol', '')
            quote = airdrop.get('futuresQuoteAsset', '')
            print(f"  {asset} -> {futures_symbol} (quoted in {quote})")

if __name__ == "__main__":
    main()