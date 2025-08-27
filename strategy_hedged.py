import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vectorbt as vbt
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

class StrategyHedged:
    def __init__(self, data_folder: str, btc_file: str, initial_portfolio: float = 10000, investment_per_asset: float = 1000):
        """
        Initialize the hedged crypto strategy: Short tokens + Long BTC.
        
        Args:
            data_folder: Path to folder containing JSON price history files for tokens
            btc_file: Path to BTC.json file
            initial_portfolio: Starting portfolio value
            investment_per_asset: Amount to invest per asset (short tokens, long BTC)
        """
        self.data_folder = data_folder
        self.btc_file = btc_file
        self.initial_portfolio = initial_portfolio
        self.investment_per_asset = investment_per_asset
        self.assets_data = {}
        self.btc_data = None
        self.trades_df = None
        self.price_df = None
        self.portfolio = None
        
        # Configure VectorBT settings
        vbt.settings.set_theme("dark")
        vbt.settings.portfolio.stats['incl_unrealized'] = True
        
    def load_asset_data(self) -> Dict[str, pd.DataFrame]:
        """Load all asset data from JSON files including BTC."""
        
        # Load token data
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
        
        # Load BTC data
        with open(self.btc_file, 'r') as f:
            btc_data = json.load(f)
        
        btc_df = pd.DataFrame(btc_data['price_data'])
        btc_df['datetime'] = pd.to_datetime(btc_df['datetime'])
        btc_df.set_index('datetime', inplace=True)
        self.btc_data = btc_df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Add BTC to assets_data for unified processing
        self.assets_data['BTC'] = self.btc_data
        
        return self.assets_data
    
    def prepare_vectorbt_data(self):
        """Prepare data for VectorBT simulation with both shorts and longs."""
        
        # Create unified price DataFrame for VectorBT
        price_data = {}
        for asset_name, asset_data in self.assets_data.items():
            price_data[asset_name] = asset_data['close']
        
        self.price_df = pd.DataFrame(price_data)
        self.price_df = self.price_df.fillna(method='ffill').fillna(method='bfill')
        
        # Create signal matrices
        short_entries = pd.DataFrame(False, index=self.price_df.index, columns=self.price_df.columns)
        short_exits = pd.DataFrame(False, index=self.price_df.index, columns=self.price_df.columns)
        long_entries = pd.DataFrame(False, index=self.price_df.index, columns=self.price_df.columns)
        long_exits = pd.DataFrame(False, index=self.price_df.index, columns=self.price_df.columns)
        
        # Track when each token starts trading (first non-null price)
        token_start_dates = {}
        
        for asset in self.price_df.columns:
            if asset != 'BTC':
                # Find first valid price for this token
                first_valid_idx = self.price_df[asset].first_valid_index()
                if first_valid_idx is not None:
                    token_start_dates[asset] = first_valid_idx
                    
                    # Short the token at its first candle
                    short_entries.loc[first_valid_idx, asset] = True
                    # Exit short at last candle
                    short_exits.iloc[-1][asset] = True
                    
                    # Long BTC at the same time as shorting the token
                    long_entries.loc[first_valid_idx, 'BTC'] = True
        
        # Exit all BTC longs at the end
        long_exits.iloc[-1]['BTC'] = True
        
        return short_entries, short_exits, long_entries, long_exits, token_start_dates
    
    def run_vectorbt_simulation(self):
        """Run VectorBT simulation with both shorts and longs."""
        
        # Prepare signals
        short_entries, short_exits, long_entries, long_exits, token_start_dates = self.prepare_vectorbt_data()
        
        # For VectorBT: 
        # - entries/exits = long positions
        # - short_entries/short_exits = short positions
        # - Use positive size values for both
        
        # Create size matrices (positive values for both)
        long_size = long_entries * self.investment_per_asset
        short_size = short_entries * self.investment_per_asset
        
        # Run VectorBT simulation with correct parameters
        self.portfolio = vbt.Portfolio.from_signals(
            close=self.price_df,
            entries=long_entries,          # Long entries (BTC)
            exits=long_exits,              # Long exits (BTC)
            short_entries=short_entries,   # Short entries (tokens)
            short_exits=short_exits,       # Short exits (tokens)
            size=long_size,                # Size for long positions
            size_type='value',
            init_cash=self.initial_portfolio,
            fees=0.001,
            freq='D',
        )
        
        self.token_start_dates = token_start_dates
        return self.portfolio
    
    def extract_trades_from_vectorbt(self):
        """Extract trade data from VectorBT portfolio."""
        
        try:
            trades_readable = self.portfolio.trades.records_readable
            
            if len(trades_readable) == 0:
                return self.create_fallback_trades()
            
            trades_list = []
            
            for _, trade in trades_readable.iterrows():
                asset = trade['Column']
                entry_price = trade['Entry Price']
                exit_price = trade['Exit Price'] 
                size = trade['Size']
                pnl = trade['PnL']
                return_pct = trade['Return [%]']
                
                # Determine if this is a short or long based on size sign
                trade_type = 'SHORT' if size < 0 else 'LONG'
                quantity = abs(size)
                investment = quantity * entry_price
                
                trades_list.append({
                    'trade_id': f"{trade_type}_{asset}",
                    'asset': asset,
                    'trade_type': trade_type,
                    'entry_date': trade['Entry Timestamp'],
                    'exit_date': trade['Exit Timestamp'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'investment': investment,
                    'pnl': pnl,
                    'return_pct': return_pct,
                    'side': -1 if trade_type == 'SHORT' else 1,
                    'cash_flow': investment if trade_type == 'SHORT' else -investment
                })
            
            self.trades_df = pd.DataFrame(trades_list)
            return self.trades_df
            
        except Exception as e:
            return self.create_fallback_trades()
    
    def create_fallback_trades(self):
        """Fallback method if VectorBT trade extraction fails."""
        
        trades_list = []
        btc_long_count = 0
        
        for asset_name, price_data in self.assets_data.items():
            if asset_name == 'BTC':
                continue  # Skip BTC for now, handle separately
                
            # Token SHORT trade
            entry_date = price_data.index[0]
            exit_date = price_data.index[-1]
            entry_price = price_data['close'].iloc[0]
            exit_price = price_data['close'].iloc[-1]
            tokens_shorted = self.investment_per_asset / entry_price
            
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
                'cash_flow': self.investment_per_asset
            })
            
            # Corresponding BTC LONG trade (opened at same time as token short)
            if 'BTC' in self.assets_data:
                btc_data = self.assets_data['BTC']
                # Find BTC price at token entry date
                btc_entry_date = entry_date
                if btc_entry_date in btc_data.index:
                    btc_entry_price = btc_data.loc[btc_entry_date, 'close']
                else:
                    # Find closest date
                    closest_idx = btc_data.index.get_indexer([btc_entry_date], method='nearest')[0]
                    if 0 <= closest_idx < len(btc_data):
                        btc_entry_price = btc_data.iloc[closest_idx]['close']
                        btc_entry_date = btc_data.index[closest_idx]
                    else:
                        continue
                
                btc_exit_price = btc_data['close'].iloc[-1]
                btc_exit_date = btc_data.index[-1]
                btc_bought = self.investment_per_asset / btc_entry_price
                
                btc_long_count += 1
                btc_pnl = (btc_exit_price - btc_entry_price) * btc_bought
                btc_return_pct = (btc_pnl / self.investment_per_asset) * 100
                
                trades_list.append({
                    'trade_id': f"LONG_BTC_{btc_long_count}",
                    'asset': 'BTC',
                    'trade_type': 'LONG',
                    'entry_date': btc_entry_date,
                    'exit_date': btc_exit_date,
                    'entry_price': btc_entry_price,
                    'exit_price': btc_exit_price,
                    'quantity': btc_bought,
                    'investment': self.investment_per_asset,
                    'pnl': btc_pnl,
                    'return_pct': btc_return_pct,
                    'side': 1,
                    'cash_flow': -self.investment_per_asset
                })
        
        self.trades_df = pd.DataFrame(trades_list)
        return self.trades_df
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive strategy metrics with corrected calculations."""
        
        # Get VectorBT stats
        try:
            stats = self.portfolio.stats()
        except:
            stats = {}
        
        def safe_extract(key, default=0):
            try:
                val = stats.get(key, default)
                # Handle inf and nan values
                if np.isinf(val) or np.isnan(val):
                    return default
                return float(val)
            except (ValueError, TypeError, KeyError):
                return default
        
        # Portfolio metrics with inf/nan handling
        total_return = safe_extract('Total Return [%]')
        sharpe_ratio = safe_extract('Sharpe Ratio')
        max_drawdown = safe_extract('Max Drawdown [%]')
        calmar_ratio = safe_extract('Calmar Ratio')
        win_rate = safe_extract('Win Rate [%]')
        profit_factor = safe_extract('Profit Factor', 1.0)
        
        # Manual Sharpe calculation if VectorBT gives inf
        if np.isinf(sharpe_ratio) or sharpe_ratio == 0:
            try:
                returns = self.portfolio.returns()
                if len(returns.shape) > 1:
                    daily_returns = returns.sum(axis=1).dropna()
                else:
                    daily_returns = returns.dropna()
                
                if len(daily_returns) > 1:
                    mean_return = daily_returns.mean()
                    std_return = daily_returns.std()
                    if std_return > 0:
                        sharpe_ratio = (mean_return / std_return) * np.sqrt(365)
                    else:
                        sharpe_ratio = 0.0
                else:
                    sharpe_ratio = 0.0
            except:
                sharpe_ratio = 0.0
        
        # Calculate net P&L from actual trades (ignore final portfolio value)
        if self.trades_df is not None and len(self.trades_df) > 0:
            short_trades = self.trades_df[self.trades_df['trade_type'] == 'SHORT']
            long_trades = self.trades_df[self.trades_df['trade_type'] == 'LONG']
            
            short_pnl = short_trades['pnl'].sum() if len(short_trades) > 0 else 0
            long_pnl = long_trades['pnl'].sum() if len(long_trades) > 0 else 0
            net_profit = short_pnl + long_pnl
            
            # Calculate total return based on actual investment
            total_investment = len(short_trades) * self.investment_per_asset + len(long_trades) * self.investment_per_asset
            if total_investment > 0:
                total_return = (net_profit / total_investment) * 100
        else:
            # Fallback: use portfolio value but calculate properly
            portfolio_value = self.portfolio.value()
            if len(portfolio_value.shape) > 1:
                final_value = portfolio_value.sum(axis=1).iloc[-1]
            else:
                final_value = portfolio_value.iloc[-1]
            
            net_profit = final_value - self.initial_portfolio
            short_pnl = 0
            long_pnl = 0
            
            # Recalculate total return
            if self.initial_portfolio > 0:
                total_return = (net_profit / self.initial_portfolio) * 100
        
        # Calculate volatility manually
        try:
            returns = self.portfolio.returns()
            if len(returns.shape) > 1:
                daily_returns = returns.sum(axis=1).dropna()
            else:
                daily_returns = returns.dropna()
                
            volatility = daily_returns.std() * np.sqrt(365) * 100 if len(daily_returns) > 0 else 0
            
            # Handle inf/nan
            if np.isinf(volatility) or np.isnan(volatility):
                volatility = 0
        except:
            volatility = 0
        
        # Trade count
        try:
            total_trades = len(self.portfolio.trades.records) if hasattr(self.portfolio.trades, 'records') else 0
        except:
            total_trades = len(self.trades_df) if self.trades_df is not None else 0
        
        # Recovery factor
        if abs(max_drawdown) > 0.01:  # Avoid division by very small numbers
            recovery_factor = abs(net_profit / (max_drawdown * self.initial_portfolio / 100))
        else:
            recovery_factor = 0
        
        # Handle inf/nan in recovery factor
        if np.isinf(recovery_factor) or np.isnan(recovery_factor):
            recovery_factor = 0
        
        # Final value for display (but we care about net_profit)
        try:
            portfolio_value = self.portfolio.value()
            if len(portfolio_value.shape) > 1:
                final_value = portfolio_value.sum(axis=1).iloc[-1]
            else:
                final_value = portfolio_value.iloc[-1]
        except:
            final_value = self.initial_portfolio + net_profit
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'volatility': volatility,
            'recovery_factor': recovery_factor,
            'total_trades': total_trades,
            'net_profit': net_profit,  # This is what matters
            'final_value': final_value,  # Just for display
            'short_pnl': short_pnl,
            'long_pnl': long_pnl
        }
    
    def create_hedged_strategy_chart(self, metrics: Dict):
        """Create comprehensive chart showing the hedged strategy performance."""
        
        os.makedirs("./outputs", exist_ok=True)
        
        # Get portfolio values
        portfolio_value = self.portfolio.value()
        if len(portfolio_value.shape) > 1:
            portfolio_values = portfolio_value.sum(axis=1)
        else:
            portfolio_values = portfolio_value
        
        # Calculate PNL timeline
        baseline_value = portfolio_values.iloc[0]
        total_pnl = portfolio_values - baseline_value
        
        # Create figure
        fig = plt.figure(figsize=(20, 14))
        
        # Main combined PNL chart
        ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=3)
        
        # Plot combined strategy only
        ax1.plot(total_pnl.index, total_pnl.values, 
                linewidth=4, color='purple', label='HEDGED Strategy (SHORT Tokens + LONG BTC)', alpha=0.9)
        
        # Break even line
        ax1.axhline(y=0, color='blue', linestyle='-', alpha=0.8, linewidth=2, label='Break Even')
        
        # Add drawdown visualization
        running_max = portfolio_values.expanding().max()
        drawdown_dollars = portfolio_values - running_max
        
        ax1.fill_between(total_pnl.index, total_pnl.values, 
                        total_pnl.values - drawdown_dollars, 
                        where=(drawdown_dollars < 0), color='darkred', alpha=0.2, label='Drawdown')
        
        # Add trade entry markers with better labeling
        if self.trades_df is not None and len(self.trades_df) > 0:
            plotted_short = False
            plotted_long = False
            
            for _, trade in self.trades_df.iterrows():
                entry_date = trade['entry_date']
                trade_type = trade['trade_type']
                asset_name = trade['asset']

                if entry_date in total_pnl.index:
                    entry_pnl = total_pnl.loc[entry_date]
                else:
                    closest_idx = total_pnl.index.get_indexer([entry_date], method='nearest')[0]
                    if closest_idx >= 0 and closest_idx < len(total_pnl):
                        entry_pnl = total_pnl.iloc[closest_idx]
                    else:
                        continue

                if trade_type == 'SHORT':
                    if not plotted_short:
                        ax1.scatter(entry_date, entry_pnl, color='red', s=120,
                                   marker='v', zorder=5, edgecolors='black', linewidth=1.5, 
                                   label='Token SHORT Entry')
                        plotted_short = True
                    else:
                        ax1.scatter(entry_date, entry_pnl, color='red', s=100,
                                   marker='v', zorder=5, edgecolors='black', linewidth=1, alpha=0.8)
                    
                    # Add token name label for shorts
                    ax1.annotate(f"S-{asset_name}",
                                 xy=(entry_date, entry_pnl),
                                 xytext=(0, -20),
                                 textcoords='offset points',
                                 ha='center',
                                 fontsize=9,
                                 fontweight='bold',
                                 color='darkred',
                                 bbox=dict(boxstyle='round,pad=0.2', 
                                          facecolor='lightcoral', 
                                          alpha=0.9, edgecolor='darkred'))
                
                elif trade_type == 'LONG':
                    if not plotted_long:
                        ax1.scatter(entry_date, entry_pnl, color='green', s=120,
                                   marker='^', zorder=5, edgecolors='black', linewidth=1.5, 
                                   label='BTC LONG Entry')
                        plotted_long = True
                    else:
                        ax1.scatter(entry_date, entry_pnl, color='green', s=100,
                                   marker='^', zorder=5, edgecolors='black', linewidth=1, alpha=0.8)
                    
                    # Add BTC label for longs
                    ax1.annotate(f"L-BTC",
                                 xy=(entry_date, entry_pnl),
                                 xytext=(0, 20),
                                 textcoords='offset points',
                                 ha='center',
                                 fontsize=9,
                                 fontweight='bold',
                                 color='darkgreen',
                                 bbox=dict(boxstyle='round,pad=0.2', 
                                          facecolor='lightgreen', 
                                          alpha=0.9, edgecolor='darkgreen'))
        
        # Styling
        ax1.set_title('HEDGED STRATEGY: SHORT Tokens + LONG BTC (VectorBT)', 
                     fontsize=20, fontweight='bold', pad=20)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Cumulative PNL ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9, loc='upper left')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Performance metrics table
        ax2 = plt.subplot2grid((4, 3), (3, 0), colspan=3)
        ax2.axis('off')
        
        metrics_data = [
            ['Total Return', f"{metrics['total_return']:.2f}%"],
            ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.3f}" if not np.isinf(metrics['sharpe_ratio']) else "N/A"],
            ['Max Drawdown', f"{metrics['max_drawdown']:.2f}%"],
            ['Calmar Ratio', f"{metrics['calmar_ratio']:.3f}" if not np.isinf(metrics['calmar_ratio']) else "N/A"],
            ['Win Rate', f"{metrics['win_rate']:.1f}%"],
            ['Profit Factor', f"{metrics['profit_factor']:.2f}"],
            ['Volatility (Ann.)', f"{metrics['volatility']:.2f}%"],
            ['Recovery Factor', f"{metrics['recovery_factor']:.2f}" if not np.isinf(metrics['recovery_factor']) else "N/A"],
            ['Net P&L', f"${metrics['net_profit']:,.2f}"],
            ['SHORT P&L', f"${metrics['short_pnl']:,.2f}"],
            ['LONG P&L', f"${metrics['long_pnl']:,.2f}"]
        ]
        
        table = ax2.table(cellText=metrics_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.25, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Color coding for metrics
        for i in range(len(metrics_data)):
            metric_name = metrics_data[i][0]
            if 'Return' in metric_name and metrics['total_return'] > 0:
                table[(i+1, 1)].set_facecolor('#90EE90')
            elif 'Profit' in metric_name and metrics['net_profit'] > 0:
                table[(i+1, 1)].set_facecolor('#90EE90')
            elif 'Drawdown' in metric_name and metrics['max_drawdown'] > -10:
                table[(i+1, 1)].set_facecolor('#90EE90')
            elif ('Sharpe' in metric_name and metrics['sharpe_ratio'] > 1) or ('Calmar' in metric_name and metrics['calmar_ratio'] > 1):
                table[(i+1, 1)].set_facecolor('#90EE90')
            elif 'Return' in metric_name or 'Profit' in metric_name or 'Drawdown' in metric_name or 'Sharpe' in metric_name or 'Calmar' in metric_name:
                table[(i+1, 1)].set_facecolor('#FFB6C1')
        
        # Header styling
        for j in range(2):
            table[(0, j)].set_facecolor('#4CAF50')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        
        # Save chart
        chart_filename = "./outputs/strategy_hedged.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()
        return fig
    
    def run_hedged_strategy(self):
        """Run the complete hedged strategy."""
        
        # Step 1: Load data
        self.load_asset_data()
        
        # Step 2: Run VectorBT simulation
        self.run_vectorbt_simulation()
        
        # Step 3: Extract trades
        self.extract_trades_from_vectorbt()
        
        # Step 4: Calculate metrics
        metrics = self.calculate_metrics()
        
        # Step 5: Create chart
        fig = self.create_hedged_strategy_chart(metrics)
        
        return metrics, self.trades_df, self.portfolio

# Usage
if __name__ == "__main__":
    strategy = StrategyHedged(
        data_folder="./outputs/prices_history",
        btc_file="./outputs/BTC.json",
        initial_portfolio=10000,
        investment_per_asset=1000
    )
    
    metrics, trades_df, portfolio = strategy.run_hedged_strategy()