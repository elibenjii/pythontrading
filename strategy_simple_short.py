import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CryptoPortfolioShortStrategy:
    def __init__(self, data_folder: str, initial_portfolio: float = 10000, investment_per_asset: float = 1000):
        """
        Initialize the crypto portfolio SHORT strategy with vectorized calculations.
        
        Args:
            data_folder: Path to folder containing JSON price history files
            initial_portfolio: Starting portfolio value
            investment_per_asset: Amount to invest (short) in each asset
        """
        self.data_folder = data_folder
        self.initial_portfolio = initial_portfolio
        self.investment_per_asset = investment_per_asset
        self.assets_data = {}
        self.trades_df = None  # Store all SHORT trades in vectorized format
        
    def load_asset_data(self) -> Dict[str, pd.DataFrame]:
        """Load all asset data from JSON files."""
        print("Loading asset data for SHORT strategy...")
        
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
                print(f"Loaded {asset_name}: {len(price_df)} candles")
        
        return self.assets_data
    
    def create_trades_dataframe(self):
        """Create a vectorized trades dataframe for all SHORT positions."""
        print("\nCreating vectorized SHORT trades dataframe...")
        print("=" * 80)
        
        trades_list = []
        
        # Create SHORT trades for each asset
        for asset_name, price_data in self.assets_data.items():
            entry_date = price_data.index[0]
            exit_date = price_data.index[-1]
            entry_price = price_data['close'].iloc[0]
            exit_price = price_data['close'].iloc[-1]
            tokens_shorted = self.investment_per_asset / entry_price
            
            # SHORT P&L: profit when price goes DOWN
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
                'side': -1,  # Short = -1
                'cash_received': self.investment_per_asset  # Cash from short sale
            })
            
            print(f"{asset_name:12} SHORT: {tokens_shorted:.6f} tokens @ ${entry_price:.4f} -> ${exit_price:.4f}")
            print(f"{'':12} Cash received: ${self.investment_per_asset:.2f} | Final PNL: ${short_pnl:.2f} ({short_return_pct:.1f}%)")
            print("-" * 80)
        
        self.trades_df = pd.DataFrame(trades_list)
        return self.trades_df
    
    def calculate_portfolio_value_timeline_vectorized(self) -> pd.DataFrame:
        """Calculate portfolio value timeline using vectorized operations."""
        print("\nCalculating SHORT portfolio timeline (vectorized)...")
        
        # Get all unique dates
        all_dates = set()
        for asset_data in self.assets_data.values():
            all_dates.update(asset_data.index)
        all_dates = sorted(list(all_dates))
        
        # Create master price DataFrame with all assets
        price_df = pd.DataFrame(index=all_dates)
        
        # Add all asset prices with forward fill for missing dates
        for asset_name, asset_data in self.assets_data.items():
            price_df[f"{asset_name}_price"] = asset_data['close'].reindex(all_dates, method='ffill')
        
        # Calculate PNL for each SHORT trade at each date using vectorization
        pnl_df = pd.DataFrame(index=all_dates)
        
        for _, trade in self.trades_df.iterrows():
            trade_id = trade['trade_id']
            asset = trade['asset']
            entry_date = trade['entry_date']
            entry_price = trade['entry_price']
            quantity = trade['quantity']
            
            price_col = f"{asset}_price"
            
            # Vectorized SHORT PNL: (entry_price - current_price) * quantity
            # Profit when current_price < entry_price (price went down)
            mask = price_df.index >= entry_date
            pnl_df.loc[mask, trade_id] = (entry_price - price_df.loc[mask, price_col]) * quantity
            pnl_df.loc[~mask, trade_id] = 0  # No PNL before entry
        
        # Fill NaN values with 0
        pnl_df = pnl_df.fillna(0)
        
        # Calculate portfolio timeline using vectorized sum
        portfolio_timeline = pd.DataFrame(index=all_dates)
        portfolio_timeline['Total_PNL'] = pnl_df.sum(axis=1)  # Sum all SHORT PNLs
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
        print("TRADING BOT PERFORMANCE METRICS")
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
        print("📊 KEY TRADING BOT METRICS:")
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
        """Calculate metrics based on individual SHORT trades."""
        print(f"\n{'='*90}")
        print("TRADE-BASED SHORT PERFORMANCE ANALYSIS")
        print(f"{'='*90}")
        
        # Calculate trade statistics using vectorized operations
        total_trades = len(self.trades_df)
        winning_trades = (self.trades_df['pnl'] > 0).sum()
        losing_trades = (self.trades_df['pnl'] <= 0).sum()
        win_rate = (self.trades_df['pnl'] > 0).mean() * 100
        
        total_pnl = self.trades_df['pnl'].sum()
        total_investment = self.trades_df['investment'].sum()
        avg_return_per_trade = self.trades_df['return_pct'].mean()
        
        # Winning and losing trade statistics
        winning_mask = self.trades_df['pnl'] > 0
        losing_mask = self.trades_df['pnl'] <= 0
        
        avg_win = self.trades_df[winning_mask]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = self.trades_df[losing_mask]['pnl'].mean() if losing_trades > 0 else 0
        best_trade = self.trades_df['pnl'].max()
        worst_trade = self.trades_df['pnl'].min()
        
        # Risk metrics
        total_cash_received = self.trades_df['cash_received'].sum()
        overall_return_pct = (total_pnl / total_investment) * 100
        profit_factor = abs(avg_win * winning_trades) / abs(avg_loss * losing_trades) if losing_trades > 0 else float('inf')
        
        print("SHORT TRADES PERFORMANCE:")
        print("-" * 50)
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades} ({win_rate:.1f}%)")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"")
        print(f"Total Investment: ${total_investment:,.2f}")
        print(f"Total Cash Received: ${total_cash_received:,.2f}")
        print(f"Total PNL: ${total_pnl:,.2f}")
        print(f"Overall Return: {overall_return_pct:.2f}%")
        print(f"Average Return per Trade: {avg_return_per_trade:.2f}%")
        print(f"")
        print(f"Best Trade: ${best_trade:,.2f}")
        print(f"Worst Trade: ${worst_trade:,.2f}")
        print(f"Average Win: ${avg_win:,.2f}")
        print(f"Average Loss: ${avg_loss:,.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        
        # Price movement analysis
        print(f"\nPRICE MOVEMENT ANALYSIS:")
        print("-" * 50)
        
        # Calculate price changes vectorized
        self.trades_df['price_change_pct'] = ((self.trades_df['exit_price'] - self.trades_df['entry_price']) / self.trades_df['entry_price']) * 100
        self.trades_df['price_direction'] = self.trades_df['price_change_pct'].apply(lambda x: 'DOWN' if x < 0 else 'UP' if x > 0 else 'FLAT')
        
        price_down_count = (self.trades_df['price_change_pct'] < 0).sum()
        price_up_count = (self.trades_df['price_change_pct'] > 0).sum()
        avg_price_change = self.trades_df['price_change_pct'].mean()
        
        print(f"Assets that went DOWN: {price_down_count} ({(price_down_count/total_trades)*100:.1f}%)")
        print(f"Assets that went UP: {price_up_count} ({(price_up_count/total_trades)*100:.1f}%)")
        print(f"Average Price Change: {avg_price_change:.2f}%")
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_investment': total_investment,
            'overall_return_pct': overall_return_pct,
            'avg_return_per_trade': avg_return_per_trade,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'profit_factor': profit_factor,
            'price_down_count': price_down_count,
            'avg_price_change': avg_price_change
        }
    
    def create_individual_trades_analysis(self):
        """Display individual SHORT trade performance with vectorized sorting."""
        print(f"\n{'='*120}")
        print("INDIVIDUAL SHORT TRADES PERFORMANCE")
        print(f"{'='*120}")
        
        # Sort trades by PNL descending using vectorized operations
        sorted_trades = self.trades_df.sort_values('pnl', ascending=False)
        
        print(f"{'TRADE ID':<20} {'ASSET':<10} {'ENTRY $':<10} {'EXIT $':<10} {'QTY':<12} {'PRICE Δ%':<10} {'PNL $':<12} {'RETURN %':<10}")
        print("-" * 120)
        
        for _, trade in sorted_trades.iterrows():
            price_change = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100
            print(f"{trade['trade_id']:<20} {trade['asset']:<10} {trade['entry_price']:<10.4f} "
                  f"{trade['exit_price']:<10.4f} {trade['quantity']:<12.6f} {price_change:<10.1f} "
                  f"{trade['pnl']:<12.2f} {trade['return_pct']:<10.2f}")
        
        # Summary statistics
        profitable_trades = sorted_trades[sorted_trades['pnl'] > 0]
        losing_trades = sorted_trades[sorted_trades['pnl'] <= 0]
        
        if len(profitable_trades) > 0:
            print(f"\n🟢 TOP 3 PROFITABLE SHORTS (Price went DOWN):")
            for _, trade in profitable_trades.head(3).iterrows():
                price_drop = ((trade['entry_price'] - trade['exit_price']) / trade['entry_price']) * 100
                print(f"   {trade['asset']}: Price dropped {price_drop:.1f}% → Profit ${trade['pnl']:.2f}")
        
        if len(losing_trades) > 0:
            print(f"\n🔴 TOP 3 LOSING SHORTS (Price went UP):")
            for _, trade in losing_trades.tail(3).iterrows():
                price_rise = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100
                print(f"   {trade['asset']}: Price rose {price_rise:.1f}% → Loss ${trade['pnl']:.2f}")
    
    def create_enhanced_trading_bot_chart(self, timeline_df: pd.DataFrame, bot_metrics: Dict):
        """Create enhanced PNL chart with trading bot metrics displayed."""
        os.makedirs("./outputs", exist_ok=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 14))
        
        # Main PNL chart (top 70% of figure)
        ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=3)
        
        # Main PNL line
        ax1.plot(timeline_df.index, timeline_df['Total_PNL'], 
                linewidth=4, color='purple', label='SHORT Strategy Total PNL', alpha=0.9)
        
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
        
        # === MODIFICATION START ===
        # Add individual trade entry markers and labels
        plotted_legend_entry = False
        for _, trade in self.trades_df.iterrows():
            entry_date = trade['entry_date']
            asset_name = trade['asset']

            # Find the corresponding PNL at the entry date
            # Handle cases where the exact entry date might not be in the timeline index
            if entry_date in timeline_df.index:
                entry_pnl = timeline_df.loc[entry_date, 'Total_PNL']
            else:
                # Find the closest date available in the index if exact match not found
                closest_date_index = timeline_df.index.get_loc(entry_date, method='nearest')
                closest_date = timeline_df.index[closest_date_index]
                entry_pnl = timeline_df.loc[closest_date, 'Total_PNL']

            # Add a downward triangle marker for the short entry
            # Use a flag to add the legend entry only once to avoid clutter
            label = 'Short Entry' if not plotted_legend_entry else None
            ax1.scatter(entry_date, entry_pnl, color='orange', s=120,
                       marker='o', zorder=5, edgecolors='black', linewidth=2, label=label)



            if not plotted_legend_entry:
                plotted_legend_entry = True


            # Add text label (annotation) with the asset name
            ax1.annotate(asset_name,
                         xy=(entry_date, entry_pnl), # Point to annotate
                         xytext=(0, 10), # Offset the text 10 points vertically above the marker
                         textcoords='offset points',
                         ha='center',
                         fontsize=9,
                         fontweight='bold',
                         color='darkred',
                          arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                                alpha=0.9, edgecolor='orange'))
        # === MODIFICATION END ===
        
        # Color fill areas
        ax1.fill_between(timeline_df.index, timeline_df['Total_PNL'], 0, 
                        where=(timeline_df['Total_PNL'] >= 0), color='green', alpha=0.3, label='Profit Zone')
        ax1.fill_between(timeline_df.index, timeline_df['Total_PNL'], 0, 
                        where=(timeline_df['Total_PNL'] < 0), color='red', alpha=0.3, label='Loss Zone')
        
        ax1.set_title('STRATEGY: SHORT 1k per Asset', 
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
        chart_filename = "./outputs/strategy_simple_short.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Enhanced Trading Bot Chart saved to: {chart_filename}")
        
        # Print strategy summary
        final_pnl = timeline_df['Total_PNL'].iloc[-1]
        total_investment = len(self.trades_df) * self.investment_per_asset
        
        print("\n📍 TRADING BOT SUMMARY:")
        print(f"  • Strategy: SHORT {len(self.trades_df)} assets @ ${self.investment_per_asset:,} each")
        print(f"  • Total Capital: ${total_investment:,}")
        print(f"  • Net P&L: ${final_pnl:,.2f}")
        print(f"  • ROI: {(final_pnl/self.initial_portfolio)*100:.2f}%")
        print(f"  • Sharpe Ratio: {bot_metrics['sharpe_ratio']:.3f}")
        print(f"  • Max Drawdown: {bot_metrics['max_drawdown']:.2f}%")
        
        plt.show()
        return fig
    
    def run_short_strategy(self):
        """Run the complete vectorized SHORT trading strategy with enhanced metrics."""
        print("Starting Enhanced Crypto Portfolio SHORT Strategy with Trading Bot Metrics")
        print("=" * 90)
        print("📉 SHORT STRATEGY: Profit when crypto prices go DOWN (VECTORIZED)")
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
        
        # Show individual trades analysis
        self.create_individual_trades_analysis()
        
        # Create enhanced trading bot chart with metrics
        self.create_enhanced_trading_bot_chart(timeline_df, bot_metrics)
        
        return timeline_df, bot_metrics, trade_metrics, self.trades_df

# Usage
if __name__ == "__main__":
    strategy = CryptoPortfolioShortStrategy(
        data_folder="./outputs/prices_history",
        initial_portfolio=10000,
        investment_per_asset=1000
    )
    
    timeline_df, bot_metrics, trade_metrics, trades_df = strategy.run_short_strategy()