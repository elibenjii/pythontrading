import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CryptoPortfolioHedgeStrategy:
    def __init__(self, data_folder: str, initial_portfolio: float = 10000, 
                 investment_per_asset: float = 1000, btc_file_path: str = "./outputs/BTC.json"):
        """
        Initialize the crypto portfolio HEDGE strategy (SHORT altcoins + LONG Bitcoin).
        
        Args:
            data_folder: Path to folder containing JSON price history files for altcoins
            initial_portfolio: Starting portfolio value
            investment_per_asset: Amount to invest (short altcoins + long BTC) per asset
            btc_file_path: Full path to Bitcoin data file (outside data_folder)
        """
        self.data_folder = data_folder
        self.initial_portfolio = initial_portfolio
        self.investment_per_asset = investment_per_asset
        self.btc_file_path = btc_file_path
        self.assets_data = {}
        self.btc_data = None
        self.trades_df = None  # Store all trades in vectorized format
        
    def load_asset_data(self) -> Dict[str, pd.DataFrame]:
        """Load altcoin data from data_folder and Bitcoin data from separate file."""
        print("Loading asset data for HEDGE strategy (SHORT altcoins + LONG BTC)...")
        
        # Load Bitcoin data from separate file
        print(f"Loading Bitcoin data from: {self.btc_file_path}")
        with open(self.btc_file_path, 'r') as f:
            btc_data = json.load(f)
        
        btc_df = pd.DataFrame(btc_data['price_data'])
        btc_df['datetime'] = pd.to_datetime(btc_df['datetime'])
        btc_df.set_index('datetime', inplace=True)
        self.btc_data = btc_df[['open', 'high', 'low', 'close', 'volume']].copy()
        print(f"Loaded BITCOIN (for longs): {len(self.btc_data)} candles")
        
        # Load altcoin data from data_folder
        for filename in os.listdir(self.data_folder):
            if filename.endswith('.json'):
                asset_name = filename.replace('.json', '')
                filepath = os.path.join(self.data_folder, filename)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data['price_data'])
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                price_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
                
                self.assets_data[asset_name] = price_df
                print(f"Loaded {asset_name} (for shorting): {len(price_df)} candles")
        
        if not os.path.exists(self.btc_file_path):
            raise ValueError(f"Bitcoin data file not found at: {self.btc_file_path}")
        
        return self.assets_data
    
    def create_trades_dataframe(self):
        """Create a vectorized trades dataframe instead of manual position tracking."""
        print("\nCreating vectorized trades dataframe...")
        print("=" * 80)
        
        trades_list = []
        
        # Create SHORT trades for altcoins and corresponding LONG BTC trades
        for asset_name, price_data in self.assets_data.items():
            entry_date = price_data.index[0]
            exit_date = price_data.index[-1]
            entry_price = price_data['close'].iloc[0]
            exit_price = price_data['close'].iloc[-1]
            tokens_shorted = self.investment_per_asset / entry_price
            
            # SHORT altcoin trade
            short_pnl = (entry_price - exit_price) * tokens_shorted
            short_return_pct = (short_pnl / self.investment_per_asset) * 100
            
            trades_list.append({
                'trade_id': f"SHORT_{asset_name}",
                'asset': asset_name,
                'trade_type': 'SHORT',
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': tokens_shorted,
                'investment': self.investment_per_asset,
                'pnl': short_pnl,
                'return_pct': short_return_pct,
                'side': -1  # Short = -1
            })
            
            # Find corresponding BTC price for LONG trade
            if entry_date in self.btc_data.index:
                btc_entry_price = self.btc_data.loc[entry_date, 'close']
                btc_entry_date = entry_date
            else:
                available_btc_dates = self.btc_data.index
                btc_entry_date = min(available_btc_dates, key=lambda x: abs(x - entry_date))
                btc_entry_price = self.btc_data.loc[btc_entry_date, 'close']
            
            if exit_date in self.btc_data.index:
                btc_exit_price = self.btc_data.loc[exit_date, 'close']
                btc_exit_date = exit_date
            else:
                available_btc_dates = self.btc_data.index
                btc_exit_date = min(available_btc_dates, key=lambda x: abs(x - exit_date))
                btc_exit_price = self.btc_data.loc[btc_exit_date, 'close']
            
            btc_tokens_bought = self.investment_per_asset / btc_entry_price
            btc_pnl = (btc_exit_price - btc_entry_price) * btc_tokens_bought
            btc_return_pct = (btc_pnl / self.investment_per_asset) * 100
            
            # LONG BTC trade
            trades_list.append({
                'trade_id': f"LONG_BTC_for_{asset_name}",
                'asset': 'BTC',
                'trade_type': 'LONG',
                'entry_date': btc_entry_date,
                'exit_date': btc_exit_date,
                'entry_price': btc_entry_price,
                'exit_price': btc_exit_price,
                'quantity': btc_tokens_bought,
                'investment': self.investment_per_asset,
                'pnl': btc_pnl,
                'return_pct': btc_return_pct,
                'side': 1,  # Long = 1
                'altcoin_pair': asset_name
            })
            
            print(f"{asset_name:12} SHORT: {tokens_shorted:.6f} tokens @ ${entry_price:.4f} -> ${exit_price:.4f} | PNL: ${short_pnl:.2f}")
            print(f"{'BTC HEDGE':12} LONG:  {btc_tokens_bought:.8f} BTC @ ${btc_entry_price:.2f} -> ${btc_exit_price:.2f} | PNL: ${btc_pnl:.2f}")
            print("-" * 80)
        
        self.trades_df = pd.DataFrame(trades_list)
        return self.trades_df
    
    def calculate_portfolio_value_timeline_vectorized(self) -> pd.DataFrame:
        """Calculate portfolio value timeline using vectorized operations."""
        print("\nCalculating HEDGE portfolio timeline (vectorized)...")
        
        # Get all unique dates from both altcoins and BTC
        all_dates = set()
        for asset_data in self.assets_data.values():
            all_dates.update(asset_data.index)
        all_dates.update(self.btc_data.index)
        all_dates = sorted(list(all_dates))
        
        # Create a master price DataFrame with all assets
        price_df = pd.DataFrame(index=all_dates)
        
        # Add altcoin prices
        for asset_name, asset_data in self.assets_data.items():
            price_df[f"{asset_name}_price"] = asset_data['close'].reindex(all_dates, method='ffill')
        
        # Add BTC price
        price_df['BTC_price'] = self.btc_data['close'].reindex(all_dates, method='ffill')
        
        # Calculate PNL for each trade at each date using vectorized operations
        pnl_df = pd.DataFrame(index=all_dates)
        
        for _, trade in self.trades_df.iterrows():
            trade_id = trade['trade_id']
            asset = trade['asset']
            entry_date = trade['entry_date']
            entry_price = trade['entry_price']
            quantity = trade['quantity']
            side = trade['side']
            
            # Get price column name
            if asset == 'BTC':
                price_col = 'BTC_price'
            else:
                price_col = f"{asset}_price"
            
            # Calculate PNL vectorized: (current_price - entry_price) * quantity * side
            # For SHORT: side = -1, so profit when price goes down
            # For LONG: side = 1, so profit when price goes up
            mask = price_df.index >= entry_date
            pnl_df.loc[mask, trade_id] = (price_df.loc[mask, price_col] - entry_price) * quantity * side
            pnl_df.loc[~mask, trade_id] = 0  # No PNL before entry
        
        # Fill NaN values with 0
        pnl_df = pnl_df.fillna(0)
        
        # Calculate totals using vectorized sum
        portfolio_timeline = pd.DataFrame(index=all_dates)
        portfolio_timeline['Short_PNL'] = pnl_df[[col for col in pnl_df.columns if 'SHORT' in col]].sum(axis=1)
        portfolio_timeline['Long_PNL'] = pnl_df[[col for col in pnl_df.columns if 'LONG' in col]].sum(axis=1)
        portfolio_timeline['Total_PNL'] = portfolio_timeline['Short_PNL'] + portfolio_timeline['Long_PNL']
        portfolio_timeline['Portfolio_Value'] = self.initial_portfolio + portfolio_timeline['Total_PNL']
        
        # Calculate daily returns for metrics calculation
        portfolio_timeline['Daily_Return'] = portfolio_timeline['Portfolio_Value'].pct_change()
        portfolio_timeline['Cumulative_Return'] = (portfolio_timeline['Portfolio_Value'] / self.initial_portfolio - 1) * 100
        
        # Add individual trade PNLs for analysis
        for trade_id in pnl_df.columns:
            portfolio_timeline[f"PNL_{trade_id}"] = pnl_df[trade_id]
        
        return portfolio_timeline
    
    def calculate_trading_bot_metrics(self, timeline_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive trading bot metrics."""
        print(f"\n{'='*90}")
        print("TRADING BOT PERFORMANCE METRICS - HEDGE STRATEGY")
        print(f"{'='*90}")
        
        # Basic returns
        daily_returns = timeline_df['Daily_Return'].dropna()
        portfolio_values = timeline_df['Portfolio_Value']
        total_return = (portfolio_values.iloc[-1] / self.initial_portfolio - 1) * 100
        
        # Calculate metrics
        metrics = {}
        
        # 1. Sharpe Ratio (assuming 0% risk-free rate, annualized)
        if len(daily_returns) > 0 and daily_returns.std() != 0:
            trading_days_per_year = 365  # Crypto trades 365 days
            sharpe_ratio = (daily_returns.mean() * trading_days_per_year) / (daily_returns.std() * np.sqrt(trading_days_per_year))
        else:
            sharpe_ratio = 0
        
        # 2. Maximum Drawdown
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # 3. Maximum Drawdown Duration (days)
        drawdown_start_dates = []
        drawdown_end_dates = []
        in_drawdown = False
        
        for i, dd in enumerate(drawdown):
            if dd < -0.01 and not in_drawdown:  # Start of drawdown (>0.01% down)
                drawdown_start_dates.append(timeline_df.index[i])
                in_drawdown = True
            elif dd >= -0.01 and in_drawdown:  # End of drawdown
                drawdown_end_dates.append(timeline_df.index[i])
                in_drawdown = False
        
        if in_drawdown:  # Still in drawdown at end
            drawdown_end_dates.append(timeline_df.index[-1])
        
        max_dd_duration = 0
        if len(drawdown_start_dates) > 0 and len(drawdown_end_dates) > 0:
            for start, end in zip(drawdown_start_dates, drawdown_end_dates):
                duration = (end - start).days
                max_dd_duration = max(max_dd_duration, duration)
        
        # 4. Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(365) * 100 if len(daily_returns) > 0 else 0
        
        # 5. Calmar Ratio (Annual Return / Max Drawdown)
        days_total = (timeline_df.index[-1] - timeline_df.index[0]).days
        annualized_return = (total_return / days_total) * 365 if days_total > 0 else 0
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
        
        # 6. Win Rate & Profit Factor
        winning_trades = (self.trades_df['pnl'] > 0).sum()
        total_trades = len(self.trades_df)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_wins = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].sum()
        total_losses = abs(self.trades_df[self.trades_df['pnl'] <= 0]['pnl'].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # 7. Average Trade Duration
        self.trades_df['trade_duration'] = (self.trades_df['exit_date'] - self.trades_df['entry_date']).dt.days
        avg_trade_duration = self.trades_df['trade_duration'].mean()
        
        # 8. Recovery Factor (Net Profit / Max Drawdown)
        net_profit = timeline_df['Total_PNL'].iloc[-1]
        recovery_factor = abs(net_profit / (max_drawdown * self.initial_portfolio / 100)) if max_drawdown != 0 else 0
        
        # Store metrics
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_dd_duration': max_dd_duration,
            'volatility': volatility,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_trade_duration,
            'recovery_factor': recovery_factor,
            'total_trades': total_trades,
            'net_profit': net_profit,
            'days_trading': days_total
        }
        
        # Print metrics
        print("📊 KEY HEDGE STRATEGY TRADING BOT METRICS:")
        print("-" * 50)
        print(f"Total Return: {total_return:.2f}%")
        print(f"Annualized Return: {annualized_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Max DD Duration: {max_dd_duration} days")
        print(f"Volatility (ann.): {volatility:.2f}%")
        print(f"Calmar Ratio: {calmar_ratio:.3f}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Avg Trade Duration: {avg_trade_duration:.1f} days")
        print(f"Recovery Factor: {recovery_factor:.2f}")
        print(f"Total Trades: {total_trades}")
        
        return metrics
    
    def calculate_trade_based_metrics(self):
        """Calculate metrics based on individual trades (as requested in the comment)."""
        print(f"\n{'='*90}")
        print("TRADE-BASED PERFORMANCE ANALYSIS (Individual Trade Returns)")
        print(f"{'='*90}")
        
        # Group trades by type
        short_trades = self.trades_df[self.trades_df['trade_type'] == 'SHORT'].copy()
        long_trades = self.trades_df[self.trades_df['trade_type'] == 'LONG'].copy()
        
        # Calculate trade statistics using vectorized operations
        short_stats = {
            'total_trades': len(short_trades),
            'winning_trades': (short_trades['pnl'] > 0).sum(),
            'losing_trades': (short_trades['pnl'] <= 0).sum(),
            'win_rate': (short_trades['pnl'] > 0).mean() * 100,
            'total_pnl': short_trades['pnl'].sum(),
            'avg_win': short_trades[short_trades['pnl'] > 0]['pnl'].mean() if (short_trades['pnl'] > 0).any() else 0,
            'avg_loss': short_trades[short_trades['pnl'] <= 0]['pnl'].mean() if (short_trades['pnl'] <= 0).any() else 0,
            'best_trade': short_trades['pnl'].max(),
            'worst_trade': short_trades['pnl'].min(),
            'avg_return_pct': short_trades['return_pct'].mean(),
            'total_investment': short_trades['investment'].sum()
        }
        
        long_stats = {
            'total_trades': len(long_trades),
            'winning_trades': (long_trades['pnl'] > 0).sum(),
            'losing_trades': (long_trades['pnl'] <= 0).sum(),
            'win_rate': (long_trades['pnl'] > 0).mean() * 100,
            'total_pnl': long_trades['pnl'].sum(),
            'avg_win': long_trades[long_trades['pnl'] > 0]['pnl'].mean() if (long_trades['pnl'] > 0).any() else 0,
            'avg_loss': long_trades[long_trades['pnl'] <= 0]['pnl'].mean() if (long_trades['pnl'] <= 0).any() else 0,
            'best_trade': long_trades['pnl'].max(),
            'worst_trade': long_trades['pnl'].min(),
            'avg_return_pct': long_trades['return_pct'].mean(),
            'total_investment': long_trades['investment'].sum()
        }
        
        # Print trade-based analysis
        print("SHORT ALTCOIN TRADES:")
        print("-" * 50)
        print(f"Total Trades: {short_stats['total_trades']}")
        print(f"Winning Trades: {short_stats['winning_trades']} ({short_stats['win_rate']:.1f}%)")
        print(f"Losing Trades: {short_stats['losing_trades']}")
        print(f"Total PNL: ${short_stats['total_pnl']:,.2f}")
        print(f"Average Win: ${short_stats['avg_win']:,.2f}")
        print(f"Average Loss: ${short_stats['avg_loss']:,.2f}")
        print(f"Best Trade: ${short_stats['best_trade']:,.2f}")
        print(f"Worst Trade: ${short_stats['worst_trade']:,.2f}")
        print(f"Average Return per Trade: {short_stats['avg_return_pct']:.2f}%")
        
        print("\nLONG BITCOIN TRADES:")
        print("-" * 50)
        print(f"Total Trades: {long_stats['total_trades']}")
        print(f"Winning Trades: {long_stats['winning_trades']} ({long_stats['win_rate']:.1f}%)")
        print(f"Losing Trades: {long_stats['losing_trades']}")
        print(f"Total PNL: ${long_stats['total_pnl']:,.2f}")
        print(f"Average Win: ${long_stats['avg_win']:,.2f}")
        print(f"Average Loss: ${long_stats['avg_loss']:,.2f}")
        print(f"Best Trade: ${long_stats['best_trade']:,.2f}")
        print(f"Worst Trade: ${long_stats['worst_trade']:,.2f}")
        print(f"Average Return per Trade: {long_stats['avg_return_pct']:.2f}%")
        
        # Combined statistics
        total_trades = len(self.trades_df)
        total_pnl = self.trades_df['pnl'].sum()
        total_investment = self.trades_df['investment'].sum()
        overall_return_pct = (total_pnl / total_investment) * 100
        
        print("\nCOMBINED HEDGE PERFORMANCE:")
        print("-" * 50)
        print(f"Total Trades: {total_trades}")
        print(f"Total Investment: ${total_investment:,.2f}")
        print(f"Total PNL: ${total_pnl:,.2f}")
        print(f"Overall Return: {overall_return_pct:.2f}%")
        print(f"Profit Factor: {abs(short_stats['total_pnl'] + long_stats['total_pnl']) / total_investment:.2f}")
        
        return {
            'short_stats': short_stats,
            'long_stats': long_stats,
            'total_pnl': total_pnl,
            'total_investment': total_investment,
            'overall_return_pct': overall_return_pct
        }
    
    def create_individual_trades_analysis(self):
        """Display individual trade performance."""
        print(f"\n{'='*120}")
        print("INDIVIDUAL TRADES PERFORMANCE (Trade-by-Trade Basis)")
        print(f"{'='*120}")
        
        # Sort trades by PNL descending
        sorted_trades = self.trades_df.sort_values('pnl', ascending=False)
        
        print(f"{'TRADE ID':<25} {'TYPE':<6} {'ASSET':<8} {'ENTRY $':<10} {'EXIT $':<10} {'QTY':<12} {'PNL $':<12} {'RETURN %':<10}")
        print("-" * 120)
        
        for _, trade in sorted_trades.iterrows():
            print(f"{trade['trade_id']:<25} {trade['trade_type']:<6} {trade['asset']:<8} "
                  f"{trade['entry_price']:<10.4f} {trade['exit_price']:<10.4f} {trade['quantity']:<12.6f} "
                  f"{trade['pnl']:<12.2f} {trade['return_pct']:<10.2f}")
    
    def create_enhanced_hedge_trading_bot_chart(self, timeline_df: pd.DataFrame, bot_metrics: Dict):
        """Create enhanced HEDGE PNL chart with trading bot metrics displayed."""
        os.makedirs("./outputs", exist_ok=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 14))
        
        # Main PNL chart (top 70% of figure)
        ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=3)
        
        # Main combined PNL
        ax1.plot(timeline_df.index, timeline_df['Total_PNL'], 
                linewidth=4, color='purple', label='COMBINED HEDGE PNL', alpha=0.9)
        
        # Individual components
        ax1.plot(timeline_df.index, timeline_df['Short_PNL'], 
                linewidth=2, color='red', label='SHORT Altcoins PNL', alpha=0.7, linestyle='--')
        ax1.plot(timeline_df.index, timeline_df['Long_PNL'], 
                linewidth=2, color='green', label='LONG Bitcoin PNL', alpha=0.7, linestyle='--')
        
        # Break even line
        ax1.axhline(y=0, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Break Even')
        
        # Add drawdown visualization
        portfolio_values = timeline_df['Portfolio_Value']
        running_max = portfolio_values.expanding().max()
        drawdown_dollars = portfolio_values - running_max
        
        # Fill drawdown areas
        ax1.fill_between(timeline_df.index, timeline_df['Total_PNL'], 
                        timeline_df['Total_PNL'] - drawdown_dollars, 
                        where=(drawdown_dollars < 0), color='red', alpha=0.2, label='Drawdown')
        
        # Add entry markers with individual token labels
        entry_positions = []  # Store positions to avoid overlapping labels
        
        for asset_name, asset_data in self.assets_data.items():
            entry_date = asset_data.index[0]
            if entry_date in timeline_df.index:
                entry_pnl = timeline_df.loc[entry_date, 'Total_PNL']
            else:
                closest_date = min(timeline_df.index, key=lambda x: abs(x - entry_date))
                entry_pnl = timeline_df.loc[closest_date, 'Total_PNL']
            
            # Add marker for each position entry
            ax1.scatter(entry_date, entry_pnl, color='orange', s=120, 
                       marker='o', zorder=5, edgecolors='black', linewidth=2)
            
            # Calculate label offset to avoid overlaps
            offset_y = 30
            offset_x = 10
            
            # Check for overlapping positions and adjust offset
            for prev_date, prev_pnl in entry_positions:
                if abs((entry_date - prev_date).days) < 7:  # If entries are within 7 days
                    offset_y += 40  # Stack labels vertically
                    offset_x = -offset_x  # Alternate sides
            
            entry_positions.append((entry_date, entry_pnl))
            
            # Add individual token label showing both SHORT + LONG BTC
            ax1.annotate(f'SHORT {asset_name.upper()}\n+ LONG BTC', 
                        (entry_date, entry_pnl), 
                        xytext=(offset_x, offset_y), textcoords='offset points', 
                        fontsize=9, fontweight='bold', color='darkred',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                                alpha=0.9, edgecolor='orange'),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                        ha='center')
        
        # Color fill for combined PNL
        ax1.fill_between(timeline_df.index, timeline_df['Total_PNL'], 0, 
                        where=(timeline_df['Total_PNL'] >= 0), color='green', alpha=0.3, label='Profit Zone')
        ax1.fill_between(timeline_df.index, timeline_df['Total_PNL'], 0, 
                        where=(timeline_df['Total_PNL'] < 0), color='red', alpha=0.3, label='Loss Zone')
        
        ax1.set_title('STRATEGY: SHORT 1k per Asset + LONG 1k BTC HEDGE', 
                     fontsize=18, fontweight='bold', pad=20)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Cumulative PNL ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10, loc='upper left')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Metrics display table (bottom 30% of figure)
        ax2 = plt.subplot2grid((4, 3), (3, 0), colspan=3)
        ax2.axis('off')
        
        # Prepare metrics data for table
        metrics_data = [
            ['Total Return', f"{bot_metrics['total_return']:.2f}%"],
            ['Sharpe Ratio', f"{bot_metrics['sharpe_ratio']:.3f}"],
            ['Max Drawdown', f"{bot_metrics['max_drawdown']:.2f}%"],
            ['Calmar Ratio', f"{bot_metrics['calmar_ratio']:.3f}"],
            ['Win Rate', f"{bot_metrics['win_rate']:.1f}%"],
            ['Profit Factor', f"{bot_metrics['profit_factor']:.2f}"],
            ['Volatility (Ann.)', f"{bot_metrics['volatility']:.2f}%"],
            ['Recovery Factor', f"{bot_metrics['recovery_factor']:.2f}"],
            ['Max DD Duration', f"{bot_metrics['max_dd_duration']} days"],
            ['Avg Trade Duration', f"{bot_metrics['avg_trade_duration']:.1f} days"],
            ['Total Trades', f"{bot_metrics['total_trades']}"],
            ['Net Profit', f"${bot_metrics['net_profit']:,.2f}"]
        ]
        
        # Create table
        table = ax2.table(cellText=metrics_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.3, 0.2])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Color code the table
        for i in range(len(metrics_data)):
            # Color cells based on values
            metric_name = metrics_data[i][0]
            value_str = metrics_data[i][1]
            
            if 'Return' in metric_name or 'Profit Factor' in metric_name:
                if bot_metrics['total_return'] > 0 or bot_metrics['profit_factor'] > 1:
                    table[(i+1, 1)].set_facecolor('#90EE90')  # Light green
                else:
                    table[(i+1, 1)].set_facecolor('#FFB6C1')  # Light red
            elif 'Drawdown' in metric_name:
                if bot_metrics['max_drawdown'] > -10:  # Less than 10% drawdown is good
                    table[(i+1, 1)].set_facecolor('#90EE90')
                else:
                    table[(i+1, 1)].set_facecolor('#FFB6C1')
            elif 'Sharpe' in metric_name or 'Calmar' in metric_name:
                if bot_metrics['sharpe_ratio'] > 1 or bot_metrics['calmar_ratio'] > 1:
                    table[(i+1, 1)].set_facecolor('#90EE90')
                else:
                    table[(i+1, 1)].set_facecolor('#FFB6C1')
        
        # Header styling
        for j in range(2):
            table[(0, j)].set_facecolor('#4CAF50')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        
        # Save the chart
        chart_filename = "./outputs/strategy_longbtc_shortairdrops.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Enhanced Hedge Strategy Trading Bot Chart saved to: {chart_filename}")
        
        # Print strategy summary
        final_pnl = timeline_df['Total_PNL'].iloc[-1]
        total_investment = len(self.trades_df) * self.investment_per_asset
        short_trades_count = len(self.trades_df[self.trades_df['trade_type'] == 'SHORT'])
        
        print("\n📍 HEDGE STRATEGY TRADING BOT SUMMARY:")
        print(f"  • Strategy: SHORT {short_trades_count} altcoins + LONG {short_trades_count} BTC positions @ ${self.investment_per_asset:,} each")
        print(f"  • Total Capital: ${total_investment:,}")
        print(f"  • Net P&L: ${final_pnl:,.2f}")
        print(f"  • ROI: {(final_pnl/self.initial_portfolio)*100:.2f}%")
        print(f"  • Sharpe Ratio: {bot_metrics['sharpe_ratio']:.3f}")
        print(f"  • Max Drawdown: {bot_metrics['max_drawdown']:.2f}%")
        
        plt.show()
        return fig
    
    def create_combined_pnl_chart(self, timeline_df: pd.DataFrame):
        """Create combined PNL chart showing SHORT + LONG performance."""
        os.makedirs("./outputs", exist_ok=True)
        
        plt.figure(figsize=(16, 10))
        
        # Main combined PNL
        plt.plot(timeline_df.index, timeline_df['Total_PNL'], 
                linewidth=4, color='purple', label='COMBINED HEDGE PNL', alpha=0.9)
        
        # Individual components
        plt.plot(timeline_df.index, timeline_df['Short_PNL'], 
                linewidth=2, color='red', label='SHORT Altcoins PNL', alpha=0.7, linestyle='--')
        plt.plot(timeline_df.index, timeline_df['Long_PNL'], 
                linewidth=2, color='green', label='LONG Bitcoin PNL', alpha=0.7, linestyle='--')
        
        # Break even line
        plt.axhline(y=0, color='blue', linestyle='-', alpha=0.5, label='Break Even')
        
        # Add entry markers with individual token labels
        entry_positions = []  # Store positions to avoid overlapping labels
        
        for asset_name, asset_data in self.assets_data.items():
            entry_date = asset_data.index[0]
            if entry_date in timeline_df.index:
                entry_pnl = timeline_df.loc[entry_date, 'Total_PNL']
            else:
                closest_date = min(timeline_df.index, key=lambda x: abs(x - entry_date))
                entry_pnl = timeline_df.loc[closest_date, 'Total_PNL']
            
            # Add marker for each position entry
            plt.scatter(entry_date, entry_pnl, color='orange', s=120, 
                       marker='o', zorder=5, edgecolors='black', linewidth=2)
            
            # Calculate label offset to avoid overlaps
            offset_y = 30
            offset_x = 10
            
            # Check for overlapping positions and adjust offset
            for prev_date, prev_pnl in entry_positions:
                if abs((entry_date - prev_date).days) < 7:  # If entries are within 7 days
                    offset_y += 40  # Stack labels vertically
                    offset_x = -offset_x  # Alternate sides
            
            entry_positions.append((entry_date, entry_pnl))
            
            # Add individual token label showing both SHORT + LONG BTC
            plt.annotate(f'SHORT {asset_name.upper()}\n+ LONG BTC', 
                        (entry_date, entry_pnl), 
                        xytext=(offset_x, offset_y), textcoords='offset points', 
                        fontsize=9, fontweight='bold', color='darkred',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                                alpha=0.9, edgecolor='orange'),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                        ha='center')
        
        plt.title('HEDGE STRATEGY: Vectorized Performance Analysis (Trade-Based Calculations)', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Cumulative PNL ($)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Color fill for combined PNL
        plt.fill_between(timeline_df.index, timeline_df['Total_PNL'], 0, 
                        where=(timeline_df['Total_PNL'] >= 0), color='green', alpha=0.2)
        plt.fill_between(timeline_df.index, timeline_df['Total_PNL'], 0, 
                        where=(timeline_df['Total_PNL'] < 0), color='red', alpha=0.2)
        
        plt.tight_layout()
        
        # Save the chart
        chart_filename = "./outputs/vectorized_hedge_strategy_pnl.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Vectorized Hedge Strategy Chart saved to: {chart_filename}")
        
        plt.show()
    
    def run_hedge_strategy(self):
        """Run the complete HEDGE trading strategy with vectorized calculations and enhanced metrics."""
        print("Starting Enhanced Crypto Portfolio HEDGE Strategy with Trading Bot Metrics")
        print("=" * 90)
        print("🔄 HEDGE STRATEGY: SHORT Altcoins + LONG Bitcoin (VECTORIZED)")
        print("📊 Enhanced with comprehensive trading bot performance metrics")
        print("⚡ Optimized with pandas vectorization")
        print("=" * 90)
        
        # Load data and create trades dataframe
        self.load_asset_data()
        self.create_trades_dataframe()
        
        # Calculate portfolio timeline using vectorized operations
        timeline_df = self.calculate_portfolio_value_timeline_vectorized()
        
        # Calculate comprehensive trading bot metrics
        bot_metrics = self.calculate_trading_bot_metrics(timeline_df)
        
        # Calculate trade-based metrics
        trade_metrics = self.calculate_trade_based_metrics()
        
        # Show individual trades
        self.create_individual_trades_analysis()
        
        # Create enhanced trading bot chart with metrics table
        self.create_enhanced_hedge_trading_bot_chart(timeline_df, bot_metrics)
        
        return timeline_df, bot_metrics, trade_metrics, self.trades_df

# Usage
if __name__ == "__main__":
    strategy = CryptoPortfolioHedgeStrategy(
        data_folder="./outputs/prices_history",
        initial_portfolio=10000,
        investment_per_asset=1000,
        btc_file_path="./outputs/BTC.json"
    )
    
    timeline_df, bot_metrics, trade_metrics, trades_df = strategy.run_hedge_strategy()
    