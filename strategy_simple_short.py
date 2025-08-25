import json
import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configure vectorbt settings
vbt.settings.set_theme("dark")
vbt.settings['plotting']['layout']['width'] = 1000
vbt.settings['plotting']['layout']['height'] = 600

class WiseMonkeyShortBacktester:
    def __init__(self, json_file_path):
        """
        Initialize the backtester with price data from JSON file
        """
        self.json_file_path = json_file_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """
        Load and process the JSON price data
        """
        try:
            with open(self.json_file_path, 'r') as f:
                data = json.load(f)
            
            # Extract timestamps and prices
            timestamps = [entry[0] for entry in data['stats']]
            prices = [entry[1] for entry in data['stats']]
            
            # Convert timestamps to datetime
            datetime_index = pd.to_datetime(timestamps, unit='ms')
            
            # Create DataFrame
            self.df = pd.DataFrame({
                'timestamp': datetime_index,
                'price': prices
            }).set_index('timestamp')
            
            print(f"Loaded {len(self.df)} price points")
            print(f"Date range: {self.df.index[0]} to {self.df.index[-1]}")
            print(f"Price range: {self.df['price'].min():.2e} to {self.df['price'].max():.2e}")
            
        except FileNotFoundError:
            print(f"File {self.json_file_path} not found. Using sample data.")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """
        Create sample data if the JSON file is not available
        """
        # Sample data from your example
        sample_stats = [
            [1743465872050, 1.4505292707059353e-06],
            [1743469475025, 1.4520069623740422e-06],
            [1743473074153, 1.450115172053685e-06],
            [1743476687889, 1.449082800229313e-06],
            [1743480281506, 1.4485280078367763e-06],
            # Adding more synthetic data points for better backtesting
            [1743483881506, 1.4475280078367763e-06],
            [1743487481506, 1.4495280078367763e-06],
            [1743491081506, 1.4505280078367763e-06],
            [1743494681506, 1.4515280078367763e-06],
            [1743498281506, 1.4525280078367763e-06]
        ]
        
        timestamps = [entry[0] for entry in sample_stats]
        prices = [entry[1] for entry in sample_stats]
        
        datetime_index = pd.to_datetime(timestamps, unit='ms')
        
        self.df = pd.DataFrame({
            'timestamp': datetime_index,
            'price': prices
        }).set_index('timestamp')
        
        print("Using sample data for demonstration")
    
    def calculate_indicators(self):
        """
        Calculate technical indicators for the short strategy
        """
        # Calculate returns
        self.df['returns'] = self.df['price'].pct_change()
        
        # Moving averages
        self.df['sma_short'] = self.df['price'].rolling(window=3).mean()
        self.df['sma_long'] = self.df['price'].rolling(window=5).mean()
        
        # RSI
        delta = self.df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=3).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=3).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        rolling_mean = self.df['price'].rolling(window=3).mean()
        rolling_std = self.df['price'].rolling(window=3).std()
        self.df['bb_upper'] = rolling_mean + (rolling_std * 2)
        self.df['bb_lower'] = rolling_mean - (rolling_std * 2)
        
        # Volume-based indicators (using price volatility as proxy)
        self.df['volatility'] = self.df['returns'].rolling(window=3).std()
        
        print("Technical indicators calculated")
        return self.df.head(10)
    
    def generate_short_signals(self, strategy='mean_reversion'):
        """
        Generate short signals based on different strategies
        """
        if strategy == 'mean_reversion':
            # Short when price is above upper Bollinger Band or RSI > 70
            self.df['short_signal'] = (
                (self.df['price'] > self.df['bb_upper']) | 
                (self.df['rsi'] > 70)
            )
            
        elif strategy == 'momentum':
            # Short when short MA crosses below long MA (bearish momentum)
            self.df['short_signal'] = (
                self.df['sma_short'] < self.df['sma_long']
            )
            
        elif strategy == 'volatility':
            # Short when volatility is high (expecting reversion)
            vol_threshold = self.df['volatility'].quantile(0.7)
            self.df['short_signal'] = self.df['volatility'] > vol_threshold
            
        else:  # combined strategy
            # Combine multiple signals
            mean_rev_signal = (
                (self.df['price'] > self.df['bb_upper']) | 
                (self.df['rsi'] > 70)
            )
            momentum_signal = self.df['sma_short'] < self.df['sma_long']
            vol_signal = self.df['volatility'] > self.df['volatility'].quantile(0.6)
            
            self.df['short_signal'] = mean_rev_signal | momentum_signal | vol_signal
        
        # Convert boolean to 1/-1 for vectorbt (1 = short entry, -1 = short exit)
        self.df['position'] = np.where(self.df['short_signal'], -1, 0)
        
        print(f"Generated {self.df['short_signal'].sum()} short signals using {strategy} strategy")
        return self.df[['price', 'short_signal', 'position']].head(10)
    
    def run_backtest(self, initial_cash=10000, fees=0.001):
        """
        Run a simple buy-and-hold short backtest: short from day 1 and hold
        """
        # Clean up data
        clean_df = self.df.dropna(subset=['price'])
        
        if len(clean_df) == 0:
            print("No valid price data.")
            return None
        
        print(f"Running buy-and-hold SHORT backtest with {len(clean_df)} data points")
        print(f"Entry date: {clean_df.index[0]}")
        print(f"Final date: {clean_df.index[-1]}")
        print(f"Entry price: {clean_df['price'].iloc[0]:.2e}")
        print(f"Final price: {clean_df['price'].iloc[-1]:.2e}")
        
        # Simple hold-short strategy: short from the beginning
        entry_price = clean_df['price'].iloc[0]
        
        # Calculate position size (use 95% of capital, keep 5% as cash)
        position_value = initial_cash * 0.95
        shares_shorted = position_value / entry_price  # Number of tokens we're short
        cash_remaining = initial_cash * 0.05
        
        print(f"Shares shorted: {shares_shorted:.0f}")
        print(f"Cash remaining: ${cash_remaining:.2f}")
        
        # Calculate portfolio value over time
        portfolio_values = []
        unrealized_pnl_series = []
        
        for current_price in clean_df['price']:
            # For short position: profit when price goes down
            unrealized_pnl = shares_shorted * (entry_price - current_price)
            
            # Portfolio value = cash + position_value + unrealized_pnl
            portfolio_value = cash_remaining + position_value + unrealized_pnl
            
            portfolio_values.append(portfolio_value)
            unrealized_pnl_series.append(unrealized_pnl)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'price': clean_df['price'],
            'portfolio_value': portfolio_values,
            'unrealized_pnl': unrealized_pnl_series,
            'cumulative_return': [(pv - initial_cash) / initial_cash for pv in portfolio_values]
        }, index=clean_df.index)
        
        # Create simple portfolio object for analysis
        class HoldShortPortfolio:
            def __init__(self, results_df, entry_price, shares_shorted, init_cash):
                self.results = results_df
                self.entry_price = entry_price
                self.shares_shorted = shares_shorted
                self.init_cash = init_cash
                
            def value(self):
                return self.results['portfolio_value']
                
            def total_return(self):
                return (self.results['portfolio_value'].iloc[-1] - self.init_cash) / self.init_cash
                
            def final_pnl(self):
                return self.results['unrealized_pnl'].iloc[-1]
                
            def max_drawdown(self):
                peak = self.results['portfolio_value'].expanding().max()
                drawdown = (self.results['portfolio_value'] - peak) / peak
                return drawdown.min()
                
            def daily_returns(self):
                return self.results['portfolio_value'].pct_change().dropna()
                
            def sharpe_ratio(self):
                daily_ret = self.daily_returns()
                if len(daily_ret) == 0 or daily_ret.std() == 0:
                    return 0
                return daily_ret.mean() / daily_ret.std() * np.sqrt(365)  # Annualized
        
        self.portfolio = HoldShortPortfolio(results_df, entry_price, shares_shorted, initial_cash)
        self.hold_short_results = results_df
        
        print("Buy-and-hold short backtest completed")
        return self.portfolio
    
    def analyze_performance(self, airdrop_start=None, airdrop_end=None):
        """
        Analyze performance for buy-and-hold short strategy
        """
        if not hasattr(self, 'portfolio'):
            print("Run backtest first!")
            return
        
        pf = self.portfolio
        
        # Calculate key metrics
        final_value = pf.value().iloc[-1]
        total_return = pf.total_return()
        final_pnl = pf.final_pnl()
        max_drawdown = pf.max_drawdown()
        sharpe_ratio = pf.sharpe_ratio()
        
        # Price performance
        entry_price = pf.entry_price
        final_price = self.hold_short_results['price'].iloc[-1]
        price_change = (final_price - entry_price) / entry_price
        
        # Time metrics
        start_date = self.hold_short_results.index[0]
        end_date = self.hold_short_results.index[-1]
        days_held = (end_date - start_date).days
        
        # Calculate airdrop period performance if dates provided
        airdrop_metrics = {}
        if airdrop_start and airdrop_end:
            try:
                results = self.hold_short_results
                
                print(f"\n🔍 AIRDROP DATE ANALYSIS:")
                print(f"Airdrop Start: {airdrop_start.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Airdrop End: {airdrop_end.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Price Data Start: {results.index[0].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Price Data End: {results.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Check if airdrop dates are within price data range
                if airdrop_start < results.index[0]:
                    print(f"⚠️  WARNING: Airdrop start is BEFORE price data begins!")
                    print(f"   Airdrop was {(results.index[0] - airdrop_start).days} days before data starts")
                    
                if airdrop_end < results.index[0]:
                    print(f"⚠️  WARNING: Airdrop end is BEFORE price data begins!")
                    print(f"   Entire airdrop period was {(results.index[0] - airdrop_end).days} days before data")
                    
                    airdrop_metrics = {
                        '⚠️ Airdrop Period': 'OUTSIDE PRICE DATA RANGE',
                        'Airdrop Start': airdrop_start.strftime('%Y-%m-%d'),
                        'Airdrop End': airdrop_end.strftime('%Y-%m-%d'),
                        'Price Data Starts': results.index[0].strftime('%Y-%m-%d'),
                        'Note': 'Cannot calculate airdrop price change - no data available',
                    }
                else:
                    # Find closest data points to airdrop dates
                    start_idx = results.index.get_indexer([airdrop_start], method='nearest')[0]
                    end_idx = results.index.get_indexer([airdrop_end], method='nearest')[0]
                    
                    airdrop_start_price = results.iloc[start_idx]['price']
                    airdrop_end_price = results.iloc[end_idx]['price']
                    airdrop_price_change = (airdrop_end_price - airdrop_start_price) / airdrop_start_price
                    
                    airdrop_start_portfolio = results.iloc[start_idx]['portfolio_value']
                    airdrop_end_portfolio = results.iloc[end_idx]['portfolio_value']
                    airdrop_portfolio_return = (airdrop_end_portfolio - airdrop_start_portfolio) / airdrop_start_portfolio
                    
                    airdrop_duration = (airdrop_end - airdrop_start).days
                    
                    airdrop_metrics = {
                        'Airdrop Start Date': airdrop_start.strftime('%Y-%m-%d %H:%M'),
                        'Airdrop End Date': airdrop_end.strftime('%Y-%m-%d %H:%M'),
                        'Airdrop Duration': f"{airdrop_duration} days",
                        'Price at Airdrop Start': f"{airdrop_start_price:.2e}",
                        'Price at Airdrop End': f"{airdrop_end_price:.2e}",
                        'Token Price Change (Airdrop Period)': f"{airdrop_price_change:.2%}",
                        'Portfolio Value at Airdrop Start': f"${airdrop_start_portfolio:,.2f}",
                        'Portfolio Value at Airdrop End': f"${airdrop_end_portfolio:,.2f}",
                        'Portfolio Return (Airdrop Period)': f"{airdrop_portfolio_return:.2%}",
                    }
                
            except Exception as e:
                print(f"Could not calculate airdrop metrics: {e}")
        
        metrics = {
            'Strategy': 'Buy-and-Hold SHORT',
            'Entry Date': start_date.strftime('%Y-%m-%d %H:%M'),
            'Exit Date': end_date.strftime('%Y-%m-%d %H:%M'),
            'Days Held': f"{days_held} days",
            'Entry Price': f"{entry_price:.2e}",
            'Final Price': f"{final_price:.2e}",
            'Token Price Change (Total)': f"{price_change:.2%}",
            '': '--- PORTFOLIO PERFORMANCE ---',
            'Initial Value': f"${pf.init_cash:.2f}",
            'Final Value': f"${final_value:.2f}",
            'Total Return': f"{total_return:.2%}",
            'Unrealized P&L': f"${final_pnl:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Shares Shorted': f"{pf.shares_shorted:.0f}",
        }
        
        # Add airdrop metrics if available
        if airdrop_metrics:
            metrics.update({'': '--- AIRDROP PERIOD ANALYSIS ---'})
            metrics.update(airdrop_metrics)
        
        # Find best and worst points
        results = self.hold_short_results
        best_idx = results['portfolio_value'].idxmax()
        worst_idx = results['portfolio_value'].idxmin()
        
        print("=== BUY-AND-HOLD SHORT BACKTEST RESULTS ===")
        for metric, value in metrics.items():
            if metric == '':  # Section separator
                print(value)
            else:
                print(f"{metric:.<35} {value}")
        
        # Show some key time points
        print(f"\n=== KEY TIME POINTS ===")
        
        print(f"Best performance: {best_idx.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Portfolio value: ${results.loc[best_idx, 'portfolio_value']:.2f}")
        print(f"  Token price: {results.loc[best_idx, 'price']:.2e}")
        print(f"  Unrealized P&L: ${results.loc[best_idx, 'unrealized_pnl']:.2f}")
        
        print(f"Worst performance: {worst_idx.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Portfolio value: ${results.loc[worst_idx, 'portfolio_value']:.2f}")
        print(f"  Token price: {results.loc[worst_idx, 'price']:.2e}")
        print(f"  Unrealized P&L: ${results.loc[worst_idx, 'unrealized_pnl']:.2f}")
        
        return metrics
    
    def plot_results(self, save_plots=True, output_dir='./backtest_results/', filename=None, 
                     airdrop_start=None, airdrop_end=None, show_plot=False):
        """
        Plot the buy-and-hold short backtest results and save to files
        
        Parameters:
        -----------
        airdrop_start : str or datetime, optional
            Start date of airdrop period (e.g., '2025-04-01' or datetime object)
        airdrop_end : str or datetime, optional  
            End date of airdrop period (e.g., '2025-06-29' or datetime object)
        show_plot : bool, default False
            Whether to display the plot (True) or just save it (False)
        """
        if not hasattr(self, 'portfolio'):
            print("Run backtest first!")
            return
        
        # Create output directory if it doesn't exist
        import os
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Parse airdrop dates if provided
        if airdrop_start is None:
            airdrop_start = self.hold_short_results.index[0]  # Default to backtest start
        elif isinstance(airdrop_start, str):
            airdrop_start = pd.to_datetime(airdrop_start)
            
        if airdrop_end is None:
            airdrop_end = self.hold_short_results.index[-1]  # Default to backtest end
        elif isinstance(airdrop_end, str):
            airdrop_end = pd.to_datetime(airdrop_end)
        
        # Use Agg backend to avoid opening windows
        original_backend = plt.get_backend()
        if not show_plot:
            plt.switch_backend('Agg')
        
        try:
            # Create subplots
            fig, axes = plt.subplots(4, 1, figsize=(15, 16))
            
            results = self.hold_short_results
            
            # Plot 1: Token Price Over Time
            axes[0].plot(results.index, results['price'], label='Token Price', color='blue', linewidth=2)
            axes[0].axhline(y=self.portfolio.entry_price, color='red', linestyle='--', alpha=0.7, label='Entry Price')
            
            # Add airdrop markers
            self._add_airdrop_markers(axes[0], airdrop_start, airdrop_end, results, 'price')
            
            axes[0].set_title('Token Price Over Time')
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_yscale('log')  # Log scale for small prices
            
            # Plot 2: Portfolio Value
            axes[1].plot(results.index, results['portfolio_value'], label='Portfolio Value', color='green', linewidth=2)
            axes[1].axhline(y=self.portfolio.init_cash, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
            
            # Add airdrop markers
            self._add_airdrop_markers(axes[1], airdrop_start, airdrop_end, results, 'portfolio_value')
            
            axes[1].set_title('Short Portfolio Performance')
            axes[1].set_ylabel('Portfolio Value ($)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Unrealized P&L
            axes[2].plot(results.index, results['unrealized_pnl'], label='Unrealized P&L', color='purple', linewidth=2)
            axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            axes[2].fill_between(results.index, results['unrealized_pnl'], 0, 
                               where=(results['unrealized_pnl'] > 0), color='green', alpha=0.3, label='Profit')
            axes[2].fill_between(results.index, results['unrealized_pnl'], 0, 
                               where=(results['unrealized_pnl'] < 0), color='red', alpha=0.3, label='Loss')
            
            # Add airdrop markers
            self._add_airdrop_markers(axes[2], airdrop_start, airdrop_end, results, 'unrealized_pnl')
            
            axes[2].set_title('Unrealized Profit & Loss from Short Position')
            axes[2].set_ylabel('P&L ($)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            # Plot 4: Cumulative Return
            axes[3].plot(results.index, results['cumulative_return'] * 100, label='Cumulative Return', color='orange', linewidth=2)
            axes[3].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            
            # Add airdrop markers
            self._add_airdrop_markers(axes[3], airdrop_start, airdrop_end, results, 'cumulative_return', multiply_by=100)
            
            axes[3].set_title('Short Strategy Cumulative Return')
            axes[3].set_ylabel('Return (%)')
            axes[3].set_xlabel('Time')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot with specified filename
            if save_plots:
                if filename:
                    # Use custom filename
                    if not filename.endswith('.png'):
                        filename += '.png'
                    main_filename = os.path.join(output_dir, filename)
                else:
                    # Generate timestamp for unique filename
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    main_filename = f"{output_dir}token_short_backtest_{timestamp}.png"
                
                plt.savefig(main_filename, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"Chart saved: {main_filename}")
            
            # Show plot only if requested
            if show_plot:
                plt.show()
            else:
                plt.close(fig)  # Close the figure to free memory
            
            return fig
            
        finally:
            # Restore original backend
            if not show_plot:
                plt.switch_backend(original_backend)
    
    def _add_airdrop_markers(self, ax, airdrop_start, airdrop_end, results, column, multiply_by=1):
        """
        Add airdrop start and end markers to a chart
        """
        # Find closest data points to airdrop dates
        try:
            # Get values at airdrop start and end
            start_idx = results.index.get_indexer([airdrop_start], method='nearest')[0]
            end_idx = results.index.get_indexer([airdrop_end], method='nearest')[0]
            
            start_date = results.index[start_idx]
            end_date = results.index[end_idx]
            
            start_value = results.iloc[start_idx][column] * multiply_by
            end_value = results.iloc[end_idx][column] * multiply_by
            
            # Add vertical lines for airdrop period
            ax.axvline(x=start_date, color='red', linestyle=':', alpha=0.7, linewidth=2)
            ax.axvline(x=end_date, color='red', linestyle=':', alpha=0.7, linewidth=2)
            
            # Add shaded region for airdrop period
            ax.axvspan(start_date, end_date, alpha=0.1, color='orange', label='Airdrop Period')
            
            # Add simple triangular markers and text
            # Start marker (green triangle pointing down)
            ax.scatter(start_date, start_value, marker='v', s=150, color='green', 
                      edgecolors='black', linewidths=2, zorder=10)
            
            # Start text label (positioned above/below the marker)
            offset_multiplier = 1.2 if start_value > 0 else 0.8
            ax.text(start_date, start_value * offset_multiplier, 
                   'Beginning of Airdrop\nOpening SHORT', 
                   ha='center', va='bottom' if start_value > 0 else 'top',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', 
                            edgecolor='green', alpha=0.8))
            
            # End marker (red triangle pointing up)
            ax.scatter(end_date, end_value, marker='^', s=150, color='red', 
                      edgecolors='black', linewidths=2, zorder=10)
            
            # End text label
            offset_multiplier = 1.2 if end_value > 0 else 0.8
            ax.text(end_date, end_value * offset_multiplier, 
                   'Ending of Airdrop', 
                   ha='center', va='bottom' if end_value > 0 else 'top',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', 
                            edgecolor='red', alpha=0.8))
            
        except Exception as e:
            print(f"Could not add airdrop markers: {e}")
            pass
    
    def save_summary_report(self, output_dir='./backtest_results/', filename=None):
        """
        Save comprehensive summary report to text file
        """
        if not hasattr(self, 'hold_short_results'):
            print("No results to save. Run backtest first!")
            return
        
        import os
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Use provided filename or generate default
        if filename:
            if not filename.endswith('.txt'):
                filename += '.txt'
            summary_filename = os.path.join(output_dir, filename)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_filename = f"{output_dir}backtest_summary_{timestamp}.txt"
        
        # Generate comprehensive report
        results = self.hold_short_results
        pf = self.portfolio
        
        # Find best and worst performance points
        best_idx = results['portfolio_value'].idxmax()
        worst_idx = results['portfolio_value'].idxmin()
        
        # Calculate additional metrics
        price_change = (results['price'].iloc[-1] - pf.entry_price) / pf.entry_price
        days_held = (results.index[-1] - results.index[0]).days
        
        with open(summary_filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TOKEN SHORT BACKTEST REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Strategy Overview
            f.write("STRATEGY OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write("Strategy Type: Buy-and-Hold SHORT Position\n")
            f.write("Description: Enter short position on day 1 and hold until end\n")
            f.write("Capital Allocation: 95% short position, 5% cash buffer\n\n")
            
            # Time Period
            f.write("BACKTEST PERIOD\n")
            f.write("-" * 20 + "\n")
            f.write(f"Entry Date: {results.index[0].strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Exit Date: {results.index[-1].strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {days_held} days\n")
            f.write(f"Total Data Points: {len(results)}\n\n")
            
            # Price Analysis
            f.write("TOKEN PRICE ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Entry Price: {pf.entry_price:.2e}\n")
            f.write(f"Final Price: {results['price'].iloc[-1]:.2e}\n")
            f.write(f"Token Price Change: {price_change:.2%}\n")
            f.write(f"Price Direction: {'DOWN' if price_change < 0 else 'UP'} (Favorable for SHORT: {'YES' if price_change < 0 else 'NO'})\n\n")
            
            # Portfolio Performance
            f.write("PORTFOLIO PERFORMANCE\n")
            f.write("-" * 25 + "\n")
            f.write(f"Initial Capital: ${pf.init_cash:,.2f}\n")
            f.write(f"Final Portfolio Value: ${pf.value().iloc[-1]:,.2f}\n")
            f.write(f"Total Return: {pf.total_return():.2%}\n")
            f.write(f"Absolute Profit/Loss: ${pf.final_pnl():,.2f}\n")
            f.write(f"Max Drawdown: {pf.max_drawdown():.2%}\n")
            f.write(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}\n\n")
            
            # Position Details
            f.write("POSITION DETAILS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Shares Shorted: {pf.shares_shorted:,.0f} tokens\n")
            f.write(f"Position Value at Entry: ${pf.shares_shorted * pf.entry_price:,.2f}\n")
            f.write(f"Cash Reserved: ${pf.init_cash * 0.05:,.2f}\n")
            f.write(f"Leverage Used: {(pf.shares_shorted * pf.entry_price / pf.init_cash):.1f}x\n\n")
            
            # Key Performance Points
            f.write("KEY PERFORMANCE MILESTONES\n")
            f.write("-" * 30 + "\n")
            f.write("BEST PERFORMANCE:\n")
            f.write(f"  Date: {best_idx.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Portfolio Value: ${results.loc[best_idx, 'portfolio_value']:,.2f}\n")
            f.write(f"  Token Price: {results.loc[best_idx, 'price']:.2e}\n")
            f.write(f"  Unrealized P&L: ${results.loc[best_idx, 'unrealized_pnl']:,.2f}\n")
            f.write(f"  Return at Peak: {((results.loc[best_idx, 'portfolio_value'] - pf.init_cash) / pf.init_cash):.2%}\n\n")
            
            f.write("WORST PERFORMANCE:\n")
            f.write(f"  Date: {worst_idx.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Portfolio Value: ${results.loc[worst_idx, 'portfolio_value']:,.2f}\n")
            f.write(f"  Token Price: {results.loc[worst_idx, 'price']:.2e}\n")
            f.write(f"  Unrealized P&L: ${results.loc[worst_idx, 'unrealized_pnl']:,.2f}\n")
            f.write(f"  Return at Trough: {((results.loc[worst_idx, 'portfolio_value'] - pf.init_cash) / pf.init_cash):.2%}\n\n")
            
            # Risk Analysis
            f.write("RISK ANALYSIS\n")
            f.write("-" * 15 + "\n")
            volatility = results['cumulative_return'].std() * 100
            f.write(f"Portfolio Volatility: {volatility:.2f}%\n")
            f.write(f"Maximum Loss: ${min(results['unrealized_pnl']):,.2f}\n")
            f.write(f"Maximum Gain: ${max(results['unrealized_pnl']):,.2f}\n")
            f.write(f"Risk/Reward Ratio: {abs(min(results['unrealized_pnl']) / max(results['unrealized_pnl'])):.2f}\n\n")
            
            # Monthly Performance Breakdown
            f.write("MONTHLY PERFORMANCE BREAKDOWN\n")
            f.write("-" * 35 + "\n")
            monthly_returns = results.groupby(results.index.to_period('M'))['portfolio_value'].agg(['first', 'last'])
            monthly_returns['return'] = (monthly_returns['last'] - monthly_returns['first']) / monthly_returns['first']
            
            for month, row in monthly_returns.iterrows():
                f.write(f"  {month}: {row['return']:.2%} (${row['first']:,.0f} → ${row['last']:,.0f})\n")
            
            f.write("\n")
            
            # Strategy Effectiveness
            f.write("STRATEGY EFFECTIVENESS\n")
            f.write("-" * 25 + "\n")
            win_rate = len(results[results['unrealized_pnl'] > 0]) / len(results) * 100
            f.write(f"Time in Profit: {win_rate:.1f}% of period\n")
            f.write(f"Time in Loss: {100 - win_rate:.1f}% of period\n")
            
            if pf.total_return() > 0:
                f.write(f"Strategy Result: ✓ PROFITABLE (+{pf.total_return():.2%})\n")
                f.write("Recommendation: Short strategy was successful for this token\n")
            else:
                f.write(f"Strategy Result: ✗ UNPROFITABLE ({pf.total_return():.2%})\n")
                f.write("Recommendation: Long strategy would have been better\n")
            
            f.write("\n")
            f.write("=" * 60 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 60 + "\n")
        
        print(f"Comprehensive summary saved to: {summary_filename}")
        return summary_filename

def process_multiple_tokens():
    """
    Process multiple tokens from airdrop data
    """
    import json
    import os
    
    # Load airdrop data
    try:
        with open('data_airdrop_normalized.json', 'r') as f:
            airdrop_data = json.load(f)
    except FileNotFoundError:
        print("❌ data_airdrop_normalized.json not found!")
        return
    
    # Process first 2 tokens as requested
    tokens_to_process = airdrop_data[:2]
    results_summary = []
    
    print("=" * 80)
    print("PROCESSING MULTIPLE TOKENS FOR AIRDROP SHORT STRATEGIES")
    print("=" * 80)
    
    for i, token_info in enumerate(tokens_to_process, 1):
        token_id = token_info['id']
        token_symbol = token_info['symbol'] 
        token_name = token_info['name']
        airdrop_start = pd.to_datetime(token_info['airdropPeriodStart'], unit='ms')
        airdrop_end = pd.to_datetime(token_info['airdropPeriodEnd'], unit='ms')
        
        print(f"\n{'='*60}")
        print(f"TOKEN {i}/2: {token_name} ({token_symbol.upper()})")
        print(f"ID: {token_id}")
        print(f"Airdrop: {airdrop_start.strftime('%Y-%m-%d')} to {airdrop_end.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        # Check if price history file exists
        price_file = f'price_history_tokens/{token_id}.json'
        if not os.path.exists(price_file):
            print(f"❌ Price history not found: {price_file}")
            results_summary.append({
                'token': f"{token_name} ({token_symbol})",
                'status': 'No price data',
                'file': price_file
            })
            continue
        
        try:
            # Initialize backtester for this token
            print(f"\n🔄 Processing {token_name}...")
            backtester = WiseMonkeyShortBacktester(price_file)
            
            # Calculate indicators
            print("📊 Calculating technical indicators...")
            backtester.calculate_indicators()
            
            # Run backtest
            print("💰 Running SHORT strategy...")
            pf = backtester.run_backtest(initial_cash=10000, fees=0.001)
            
            if pf is None:
                print(f"❌ Backtest failed for {token_name}")
                continue
            
            # Analyze with airdrop period
            print("📈 Analyzing performance...")
            metrics = backtester.analyze_performance(airdrop_start=airdrop_start, airdrop_end=airdrop_end)
            
            # Generate unique filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_filename = f"{token_id}_short_backtest_{timestamp}.png"
            summary_filename = f"{token_id}_short_backtest_{timestamp}.txt"
            
            # Plot and save
            print("📊 Creating charts...")
            backtester.plot_results(
                save_plots=True,
                filename=chart_filename,
                airdrop_start=airdrop_start,
                airdrop_end=airdrop_end,
                show_plot=False  # Don't open/display the plots
            )
            
            # Save summary
            print("📝 Saving summary...")
            backtester.save_summary_report(filename=summary_filename)
            
            # Collect results
            final_return = pf.total_return() if hasattr(pf, 'total_return') else 0
            results_summary.append({
                'token': f"{token_name} ({token_symbol})",
                'status': 'Success',
                'return': f"{final_return:.2%}",
                'chart': chart_filename,
                'summary': summary_filename,
                'airdrop_days': (airdrop_end - airdrop_start).days
            })
            
            print(f"✅ {token_name} completed successfully!")
            print(f"   Chart: {chart_filename}")
            print(f"   Summary: {summary_filename}")
            
        except Exception as e:
            print(f"❌ Error processing {token_name}: {str(e)}")
            results_summary.append({
                'token': f"{token_name} ({token_symbol})",
                'status': f'Error: {str(e)[:50]}...',
                'return': 'N/A'
            })
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    for result in results_summary:
        print(f"\n🪙 {result['token']}")
        print(f"   Status: {result['status']}")
        if 'return' in result:
            print(f"   Return: {result.get('return', 'N/A')}")
        if 'chart' in result:
            print(f"   Chart: {result['chart']}")
            print(f"   Summary: {result['summary']}")
        if 'airdrop_days' in result:
            print(f"   Airdrop Duration: {result['airdrop_days']} days")
    
    successful_tokens = len([r for r in results_summary if r['status'] == 'Success'])
    print(f"\n🎯 Successfully processed: {successful_tokens}/{len(tokens_to_process)} tokens")
    
    return results_summary

def main():
    """
    Main function - now processes multiple tokens from airdrop data
    """
    print("🚀 Starting Multi-Token Airdrop Short Strategy Analysis...")
    
    # Process multiple tokens from the airdrop data file
    results = process_multiple_tokens()
    
    return results

# Additional utility functions
def optimize_parameters(backtester, param_ranges):
    """
    Optimize strategy parameters using vectorbt's optimization features
    """
    # Example parameter optimization for moving average periods
    short_ma_range = np.arange(2, 6)
    long_ma_range = np.arange(4, 8)
    
    results = []
    
    for short_ma in short_ma_range:
        for long_ma in long_ma_range:
            if short_ma >= long_ma:
                continue
                
            # Recalculate indicators with new parameters
            df_temp = backtester.df.copy()
            df_temp['sma_short'] = df_temp['price'].rolling(window=short_ma).mean()
            df_temp['sma_long'] = df_temp['price'].rolling(window=long_ma).mean()
            df_temp['short_signal'] = df_temp['sma_short'] < df_temp['sma_long']
            
            # Run backtest
            pf = vbt.Portfolio.from_signals(
                close=df_temp['price'],
                short_entries=df_temp['short_signal'],
                init_cash=10000,
                fees=0.001,
                direction='both'
            )
            
            results.append({
                'short_ma': short_ma,
                'long_ma': long_ma,
                'total_return': pf.total_return()[0],
                'sharpe_ratio': pf.sharpe_ratio()[0],
                'max_drawdown': pf.max_drawdown()[0]
            })
    
    # Convert to DataFrame and find best parameters
    results_df = pd.DataFrame(results)
    best_params = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    
    print("=== PARAMETER OPTIMIZATION RESULTS ===")
    print(f"Best parameters: Short MA={best_params['short_ma']}, Long MA={best_params['long_ma']}")
    print(f"Best Sharpe Ratio: {best_params['sharpe_ratio']:.3f}")
    print(f"Total Return: {best_params['total_return']:.2%}")
    
    return results_df, best_params

if __name__ == "__main__":
    # Run the main backtesting pipeline
    results = main()
    
    print("\n=== BACKTESTING COMPLETE ===")
    print("All results saved to ./backtest_results/ directory")