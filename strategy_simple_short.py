import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vectorbt as vbt
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

class StrategySimpleShort:
    def __init__(self, data_folder: str, initial_portfolio: float = 10000, investment_per_asset: float = 1000):
        """
        Initialize the crypto portfolio SHORT strategy using VectorBT.
        
        Args:
            data_folder: Path to folder containing JSON price history files
            initial_portfolio: Starting portfolio value
            investment_per_asset: Amount to invest (short) in each asset
        """
        self.data_folder = data_folder
        self.initial_portfolio = initial_portfolio
        self.investment_per_asset = investment_per_asset
        self.assets_data = {}
        self.trades_df = None
        self.price_df = None
        self.portfolio = None
        
        # Configure VectorBT settings
        vbt.settings.set_theme("dark")
        vbt.settings.portfolio.stats['incl_unrealized'] = True
        
    def load_asset_data(self) -> Dict[str, pd.DataFrame]:
        """Load all asset data from JSON files."""
        
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
        
        return self.assets_data
    
    def prepare_vectorbt_data(self):
        """Prepare data for VectorBT simulation."""
        
        # Create unified price DataFrame for VectorBT
        price_data = {}
        for asset_name, asset_data in self.assets_data.items():
            price_data[asset_name] = asset_data['close']
        
        self.price_df = pd.DataFrame(price_data)
        self.price_df = self.price_df.fillna(method='ffill').fillna(method='bfill')
        
        # Create signal matrices - short at first candle, exit at last candle
        short_entries = pd.DataFrame(False, index=self.price_df.index, columns=self.price_df.columns)
        short_exits = pd.DataFrame(False, index=self.price_df.index, columns=self.price_df.columns)
        
        # Set signals for each asset: enter at start, exit at end
        for asset in self.price_df.columns:
            if len(self.price_df.index) > 1:
                short_entries.iloc[0][asset] = True   # Enter short at first candle
                short_exits.iloc[-1][asset] = True    # Exit short at last candle
        
        return short_entries, short_exits
    
    def run_vectorbt_simulation(self):
        """Run VectorBT simulation."""
        
        # Prepare signals
        short_entries, short_exits = self.prepare_vectorbt_data()
        
        # Create size matrix based on investment amounts
        size = short_entries * self.investment_per_asset
        
        # Run VectorBT simulation
        self.portfolio = vbt.Portfolio.from_signals(
            close=self.price_df,
            short_entries=short_entries,
            short_exits=short_exits,
            size=size,
            size_type='value',
            init_cash=self.initial_portfolio,
            fees=0.001,
            freq='D',
        )
        
        return self.portfolio
    
    def extract_trades_from_vectorbt(self):
        """Extract trade data from VectorBT portfolio.trades.records_readable"""
        
        try:
            # Get readable trade records from VectorBT
            trades_readable = self.portfolio.trades.records_readable
            
            if len(trades_readable) == 0:
                return self.create_fallback_trades()
            
            # Convert VectorBT trades to our format
            trades_list = []
            
            for _, trade in trades_readable.iterrows():
                asset = trade['Column']
                entry_price = trade['Entry Price']
                exit_price = trade['Exit Price'] 
                size = trade['Size']  # This will be negative for shorts
                pnl = trade['PnL']
                return_pct = trade['Return [%]']
                
                # Calculate additional fields for compatibility
                quantity = abs(size)  # Use absolute value for display
                investment = quantity * entry_price  # Actual investment amount
                
                trades_list.append({
                    'trade_id': f"SHORT_{asset}",
                    'asset': asset,
                    'trade_type': 'SHORT',
                    'entry_date': trade['Entry Timestamp'],
                    'exit_date': trade['Exit Timestamp'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'investment': investment,
                    'pnl': pnl,
                    'return_pct': return_pct,
                    'side': -1,  # Short = -1
                    'cash_received': investment
                })
            
            self.trades_df = pd.DataFrame(trades_list)
            return self.trades_df
            
        except Exception as e:
            return self.create_fallback_trades()
    
    def create_fallback_trades(self):
        """Fallback method if VectorBT trade extraction fails."""
        
        trades_list = []
        
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
                'side': -1,
                'cash_received': self.investment_per_asset
            })
        
        self.trades_df = pd.DataFrame(trades_list)
        return self.trades_df
    
    def calculate_trading_bot_metrics_vectorbt(self) -> Dict:
        """Calculate metrics using VectorBT stats."""
        
        # Get VectorBT stats
        stats = self.portfolio.stats()
        
        # Extract metrics with fallbacks
        def safe_extract(key, default=0):
            try:
                return float(stats.get(key, default))
            except (ValueError, TypeError, KeyError):
                return default
        
        total_return = safe_extract('Total Return [%]')
        sharpe_ratio = safe_extract('Sharpe Ratio')
        max_drawdown = safe_extract('Max Drawdown [%]')
        calmar_ratio = safe_extract('Calmar Ratio')
        win_rate = safe_extract('Win Rate [%]')
        profit_factor = safe_extract('Profit Factor', 1.0)
        
        # Additional metrics
        portfolio_value = self.portfolio.value()
        if len(portfolio_value.shape) > 1:
            final_value = portfolio_value.sum(axis=1).iloc[-1]
        else:
            final_value = portfolio_value.iloc[-1]
        
        # For SHORT strategy, net_profit should be based on PNL from trades, not portfolio value
        # Portfolio value includes unused cash, so we need to calculate actual PNL
        if self.trades_df is not None and len(self.trades_df) > 0:
            net_profit = self.trades_df['pnl'].sum()
        else:
            # Fallback to portfolio-based calculation
            net_profit = final_value - self.initial_portfolio
        
        # Calculate volatility
        returns = self.portfolio.returns()
        if len(returns.shape) > 1:
            daily_returns = returns.sum(axis=1).dropna()
        else:
            daily_returns = returns.dropna()
            
        volatility = daily_returns.std() * np.sqrt(365) * 100 if len(daily_returns) > 0 else 0
        
        # Trade count from VectorBT
        try:
            total_trades = len(self.portfolio.trades.records)
        except:
            total_trades = len(self.trades_df) if self.trades_df is not None else 0
        
        # Recovery factor - use actual net profit for calculation
        recovery_factor = abs(net_profit / (max_drawdown * self.initial_portfolio / 100)) if max_drawdown != 0 else 0
        
        # Store metrics
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'volatility': volatility,
            'recovery_factor': recovery_factor,
            'total_trades': total_trades,
            'net_profit': net_profit,
            'final_value': final_value
        }
        
        return metrics
    
    def calculate_trade_based_metrics(self):
        """Calculate trade metrics using extracted VectorBT trade data."""
        
        if self.trades_df is None or len(self.trades_df) == 0:
            return {}
        
        # Use VectorBT trade data for analysis
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
        overall_return_pct = (total_pnl / total_investment) * 100 if total_investment > 0 else 0
        profit_factor_orig = abs(avg_win * winning_trades) / abs(avg_loss * losing_trades) if losing_trades > 0 else float('inf')
        
        # Price movement analysis
        self.trades_df['price_change_pct'] = ((self.trades_df['exit_price'] - self.trades_df['entry_price']) / self.trades_df['entry_price']) * 100
        self.trades_df['price_direction'] = self.trades_df['price_change_pct'].apply(lambda x: 'DOWN' if x < 0 else 'UP' if x > 0 else 'FLAT')
        
        price_down_count = (self.trades_df['price_change_pct'] < 0).sum()
        price_up_count = (self.trades_df['price_change_pct'] > 0).sum()
        avg_price_change = self.trades_df['price_change_pct'].mean()
        
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
            'profit_factor': profit_factor_orig,
            'price_down_count': price_down_count,
            'avg_price_change': avg_price_change
        }
    
    def create_individual_trades_analysis(self):
        """Individual trades analysis using VectorBT trade data."""
        
        if self.trades_df is None or len(self.trades_df) == 0:
            return
        
        # Sort trades by PNL descending
        sorted_trades = self.trades_df.sort_values('pnl', ascending=False)
        
        # Summary statistics
        profitable_trades = sorted_trades[sorted_trades['pnl'] > 0]
        losing_trades = sorted_trades[sorted_trades['pnl'] <= 0]
        
        return sorted_trades, profitable_trades, losing_trades
    
    def create_enhanced_trading_bot_chart(self, bot_metrics: Dict):
        """Create chart using VectorBT data with original styling."""
        
        os.makedirs("./outputs", exist_ok=True)
        
        # Get VectorBT portfolio value
        portfolio_value = self.portfolio.value()
        if len(portfolio_value.shape) > 1:
            portfolio_values = portfolio_value.sum(axis=1)
        else:
            portfolio_values = portfolio_value
        
        # Calculate PNL timeline from portfolio value changes (for charting)
        baseline_value = portfolio_values.iloc[0]
        pnl_timeline = portfolio_values - baseline_value
        
        # Create figure
        fig = plt.figure(figsize=(20, 14))
        
        # Main PNL chart
        ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=3)
        
        # Main PNL line
        ax1.plot(pnl_timeline.index, pnl_timeline.values, 
                linewidth=4, color='purple', label='SHORT Strategy Total PNL', alpha=0.9)
        
        # Break even line
        ax1.axhline(y=0, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Break Even')
        
        # Add drawdown visualization
        running_max = portfolio_values.expanding().max()
        drawdown_dollars = portfolio_values - running_max
        
        ax1.fill_between(pnl_timeline.index, pnl_timeline.values, 
                        pnl_timeline.values - drawdown_dollars, 
                        where=(drawdown_dollars < 0), color='red', alpha=0.2, label='Drawdown')
        
        # Add trade entry markers
        if self.trades_df is not None and len(self.trades_df) > 0:
            plotted_legend_entry = False
            for _, trade in self.trades_df.iterrows():
                entry_date = trade['entry_date']
                asset_name = trade['asset']

                # Find PNL at entry date
                if entry_date in pnl_timeline.index:
                    entry_pnl = pnl_timeline.loc[entry_date]
                else:
                    closest_idx = pnl_timeline.index.get_indexer([entry_date], method='nearest')[0]
                    if closest_idx >= 0 and closest_idx < len(pnl_timeline):
                        entry_pnl = pnl_timeline.iloc[closest_idx]
                    else:
                        continue

                # Add marker
                label = 'Short Entry' if not plotted_legend_entry else None
                ax1.scatter(entry_date, entry_pnl, color='orange', s=120,
                           marker='v', zorder=5, edgecolors='black', linewidth=2, label=label)

                if not plotted_legend_entry:
                    plotted_legend_entry = True

                # Add annotation
                ax1.annotate(asset_name,
                             xy=(entry_date, entry_pnl),
                             xytext=(0, 10),
                             textcoords='offset points',
                             ha='center',
                             fontsize=9,
                             fontweight='bold',
                             color='darkred',
                             arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                                    alpha=0.9, edgecolor='orange'))
        
        # Color fill areas
        ax1.fill_between(pnl_timeline.index, pnl_timeline.values, 0, 
                        where=(pnl_timeline.values >= 0), color='green', alpha=0.3, label='Profit Zone')
        ax1.fill_between(pnl_timeline.index, pnl_timeline.values, 0, 
                        where=(pnl_timeline.values < 0), color='red', alpha=0.3, label='Loss Zone')
        
        # Styling
        ax1.set_title('VectorBT STRATEGY: SHORT 1k per Asset', 
                     fontsize=18, fontweight='bold', pad=20)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Cumulative PNL ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10, loc='upper left')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Metrics table
        ax2 = plt.subplot2grid((4, 3), (3, 0), colspan=3)
        ax2.axis('off')
        
        metrics_data = [
            ['Total Return', f"{bot_metrics['total_return']:.2f}%"],
            ['Sharpe Ratio', f"{bot_metrics['sharpe_ratio']:.3f}"],
            ['Max Drawdown', f"{bot_metrics['max_drawdown']:.2f}%"],
            ['Calmar Ratio', f"{bot_metrics['calmar_ratio']:.3f}"],
            ['Win Rate', f"{bot_metrics['win_rate']:.1f}%"],
            ['Profit Factor', f"{bot_metrics['profit_factor']:.2f}"],
            ['Volatility (Ann.)', f"{bot_metrics['volatility']:.2f}%"],
            ['Recovery Factor', f"{bot_metrics['recovery_factor']:.2f}"],
            ['Total Trades', f"{bot_metrics['total_trades']}"],
            ['Net Profit', f"${bot_metrics['net_profit']:,.2f}"]
        ]
        
        # Create and style table
        table = ax2.table(cellText=metrics_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.3, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Color coding
        for i in range(len(metrics_data)):
            if 'Return' in metrics_data[i][0] or 'Profit Factor' in metrics_data[i][0]:
                if bot_metrics['total_return'] > 0 or bot_metrics['profit_factor'] > 1:
                    table[(i+1, 1)].set_facecolor('#90EE90')
                else:
                    table[(i+1, 1)].set_facecolor('#FFB6C1')
            elif 'Drawdown' in metrics_data[i][0]:
                if bot_metrics['max_drawdown'] > -10:
                    table[(i+1, 1)].set_facecolor('#90EE90')
                else:
                    table[(i+1, 1)].set_facecolor('#FFB6C1')
            elif 'Sharpe' in metrics_data[i][0] or 'Calmar' in metrics_data[i][0]:
                if bot_metrics['sharpe_ratio'] > 1 or bot_metrics['calmar_ratio'] > 1:
                    table[(i+1, 1)].set_facecolor('#90EE90')
                else:
                    table[(i+1, 1)].set_facecolor('#FFB6C1')
        
        # Header styling
        for j in range(2):
            table[(0, j)].set_facecolor('#4CAF50')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        
        # Save chart
        chart_filename = "./outputs/strategy_simple_short.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()
        return fig
    
    def run_short_strategy(self):
        """Run the complete strategy using VectorBT with portfolio.trades.records_readable"""
        
        # Step 1: Load data
        self.load_asset_data()
        
        # Step 2: Run VectorBT simulation
        self.run_vectorbt_simulation()
        
        # Step 3: Extract trades from VectorBT (uses portfolio.trades.records_readable)
        self.extract_trades_from_vectorbt()
        
        # Step 4: Calculate metrics using VectorBT
        bot_metrics = self.calculate_trading_bot_metrics_vectorbt()
        
        # Step 5: Trade analysis using extracted VectorBT trade data
        trade_metrics = self.calculate_trade_based_metrics()
        
        # Step 6: Individual trades analysis
        self.create_individual_trades_analysis()
        
        # Step 7: Chart with VectorBT data
        fig = self.create_enhanced_trading_bot_chart(bot_metrics)
        
        return bot_metrics, trade_metrics, self.trades_df, self.portfolio

# Usage
if __name__ == "__main__":
    strategy = StrategySimpleShort(
        data_folder="./outputs/prices_history",
        initial_portfolio=10000,
        investment_per_asset=1000
    )
    
    bot_metrics, trade_metrics, trades_df, portfolio = strategy.run_short_strategy()