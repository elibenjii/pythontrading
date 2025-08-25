import json
import os

def load_json_file(filepath):
    """Load and return JSON data from file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filepath}")
        return None

def find_matching_tokens(airdrop_entry, tokens_data):
    """Find matching tokens based on name first, then symbol if no name match"""
    airdrop_name = airdrop_entry.get('airdropAssetFullName', '')
    airdrop_asset = airdrop_entry.get('airdropAsset', '').lower()
    
    name_matches = []
    symbol_matches = []
    
    for token in tokens_data:
        token_name = token.get('name', '')
        token_symbol = token.get('symbol', '').lower()
        token_id = token.get('id', '')
        
        # Check if the token name matches the airdrop name
        if token_name == airdrop_name:
            name_matches.append({
                'id': token_id,
                'name': token_name,
                'symbol': token_symbol,
                'match_type': 'name'
            })
        
        # Check if the token symbol matches the airdrop asset (lowercase)
        elif token_symbol == airdrop_asset:
            symbol_matches.append({
                'id': token_id,
                'name': token_name,
                'symbol': token_symbol,
                'match_type': 'symbol'
            })
    
    # Return name matches if found, otherwise symbol matches
    return name_matches if name_matches else symbol_matches

def main():
    # Load JSON files
    data_payloads = load_json_file('./data_payloads.json')
    tokens_id = load_json_file('./tokens_ids.json')
    
    if data_payloads is None or tokens_id is None:
        return
    
    # Ensure data_payloads is a list
    if not isinstance(data_payloads, list):
        data_payloads = [data_payloads]
    
    print("Processing airdrop entries...\n")
    
    # List to store normalized results
    normalized_results = []
    
    # Process each entry in data_payloads.json
    for i, airdrop_entry in enumerate(data_payloads):
        airdrop_id = airdrop_entry.get('airdropId', 'Unknown')
        airdrop_asset = airdrop_entry.get('airdropAsset', 'Unknown')
        airdrop_name = airdrop_entry.get('airdropAssetFullName', 'Unknown')
        
        print(f"Entry {i+1}: Airdrop ID {airdrop_id}")
        print(f"  Looking for: Name='{airdrop_name}' or Symbol='{airdrop_asset.lower()}'")
        
        # Find matching tokens
        matches = find_matching_tokens(airdrop_entry, tokens_id)
        
        if len(matches) == 0:
            print(f"  ❌ No matches found - NOT SAVED")
        elif len(matches) == 1:
            match = matches[0]
            match_type = match.get('match_type', 'unknown')
            print(f"  ✅ Found match by {match_type}: ID='{match['id']}', Name='{match['name']}', Symbol='{match['symbol']}' - SAVED")
            
            # Create normalized entry with only specific fields
            normalized_entry = {
                "airdropPeriodStart": airdrop_entry.get('airdropPeriodStart'),
                "airdropPeriodEnd": airdrop_entry.get('airdropPeriodEnd'),
                "id": match['id'],
                "symbol": match['symbol'],
                "name": match['name']
            }
            normalized_results.append(normalized_entry)
            
        else:
            match_type = matches[0].get('match_type', 'unknown')
            print(f"  🚨 ALERT: Found {len(matches)} matches by {match_type} - NOT SAVED!")
            for j, match in enumerate(matches):
                print(f"    Match {j+1}: ID='{match['id']}', Name='{match['name']}', Symbol='{match['symbol']}'")
        
        print()
    
    # Save normalized results to new JSON file
    if normalized_results:
        try:
            with open('./data_airdrop_normalized.json', 'w', encoding='utf-8') as file:
                json.dump(normalized_results, file, indent=2, ensure_ascii=False)
            print(f"✅ Saved {len(normalized_results)} normalized entries to './data_airdrop_normalized.json'")
        except Exception as e:
            print(f"❌ Error saving normalized data: {e}")
    else:
        print("❌ No valid matches found - no file created")

if __name__ == "__main__":
    main()