import pandas as pd
import numpy as np
from models.base_model import BaseModel

class MACrossModel(BaseModel):
    """
    移动平均线交叉策略
    """
    def __init__(self, db_manager, inst_id, timeframe='1h', short_window=20, long_window=50):
        """
        初始化移动平均线交叉策略
        
        参数:
            db_manager: 数据库管理器
            inst_id: 交易对ID
            timeframe: 时间周期
            short_window: 短期移动平均线窗口
            long_window: 长期移动平均线窗口
        """
        super().__init__(db_manager, inst_id, timeframe)
        self.short_window = short_window
        self.long_window = long_window
        
    def preprocess_data(self, df):
        """
        预处理数据，计算移动平均线
        
        参数:
            df: 原始数据
            
        返回:
            pandas.DataFrame: 预处理后的数据
        """
        # 计算短期移动平均线
        df[f'MA{self.short_window}'] = df['close'].rolling(window=self.short_window).mean()
        
        # 计算长期移动平均线
        df[f'MA{self.long_window}'] = df['close'].rolling(window=self.long_window).mean()
        
        # 删除NaN值
        df = df.dropna()
        
        return df
    
    def generate_signals(self, df):
        """
        生成交易信号
        1: 买入信号
        0: 持有
        -1: 卖出信号
        
        参数:
            df: 预处理后的数据
            
        返回:
            pandas.DataFrame: 带有交易信号的数据
        """
        # 初始化信号列
        df['signal'] = 0
        
        # 生成信号
        df['signal'] = np.where(
            df[f'MA{self.short_window}'] > df[f'MA{self.long_window}'], 1, 0
        )
        
        # 计算信号变化
        df['position'] = df['signal'].diff()
        
        return df
    
    def backtest(self, df, initial_capital=10000.0):
        """
        回测交易策略
        
        参数:
            df: 带有交易信号的数据
            initial_capital: 初始资金
            
        返回:
            dict: 回测结果
        """
        # 初始化回测结果
        positions = pd.DataFrame(index=df.index).fillna(0.0)
        positions['asset'] = df['signal']
        
        # 计算资产价值变化
        portfolio = positions.multiply(df['close'], axis=0)
        
        # 计算每日收益
        pos_diff = positions.diff()
        portfolio['holdings'] = (positions.multiply(df['close'], axis=0)).sum(axis=1)
        portfolio['cash'] = initial_capital - (pos_diff.multiply(df['close'], axis=0)).sum(axis=1).cumsum()
        
        # 计算总资产
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        
        # 计算回测指标
        total_return = (portfolio['total'][-1] / initial_capital) - 1
        annual_return = total_return / (len(df) / 252)
        sharpe_ratio = np.sqrt(252) * (portfolio['returns'].mean() / portfolio['returns'].std())
        max_drawdown = (portfolio['total'] / portfolio['total'].cummax() - 1).min()
        
        # 交易次数
        trades = df[df['position'] != 0].shape[0]
        
        # 胜率
        winning_trades = df[(df['position'] != 0) & (df['close'] > df['close'].shift())].shape[0]
        win_rate = winning_trades / trades if trades > 0 else 0
        
        # 整理回测结果
        results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'win_rate': win_rate,
            'portfolio': portfolio
        }
        
        return results