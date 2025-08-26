source venv/bin/activate

step1: fetch prices (no need they already in the code)

- run get_airdrops_list.py, get_btc_eth_prices.py, get_prices_airdrops.py

step2: run strategies (no need charts already in the code in /outputs)
strategy_simple_short.py: it will open short position $1k on each tokens at the start of their airdrop day, data from ./outputs/prices_history
strategy_longbtc_shortairdrops.py: same as strategy_simple_short.py, but it will open a long $1k on BTC everytime it short a token.

outputs strategy charts are in /outputs
