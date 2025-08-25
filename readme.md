source venv/bin/activate

step1:
copy data from
https://api.coingecko.com/api/v3/coins/list in tokens_ids.json
https://www.binance.com/bapi/asset/v1/friendly/asset-service/airdrop/list in data_payloads.json

then run step1_normalizing_dataset.py. It map to find the token ID, and create this file: data_airdrop_normalized.

step2:
run step2_fetch_price.py, it will use an headless browser with playwright to go on coingecko and fetch the price of each asset to ./price_history_tokens, headless browser is necessary to bypass coingecko CORS. The first date of the history fetch, will be the day at which airdrop started.

step3: run strategy_simple_short.py for backtesting
