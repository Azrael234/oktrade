import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from data.aggregator import KlineAggregator

class BaseModel(ABC):
    """
    交易模型基类
    """
    def __init__(self, db_manager, inst_id, timeframe='1h'):
        """
        初始化交易模型
        
        参数:
            db_manager: 数据库管理器
            inst_id: 交易对ID
            timeframe: 时间周期
        """
        self.db_manager = db_manager
        self.inst_id = inst_id
        self.timeframe = timeframe
        self.kline_aggregator = KlineAggregator(db_manager)
        self.logger = logging.getLogger(__name__)
        
    def get_data(self, start_time, end_time):
        """
        获取指定时间范围的数据
        
        参数:
            start_time: 开始时间
            end_time: 结束时间
            
        返回:
            pandas.DataFrame: 数据
        """
        # 获取合成后的K线数据
        df = self.kline_aggregator.get_aggregated_kline(
            self.inst_id, start_time, end_time, self.timeframe
        )
        
        if df.empty:
            self.logger.warning(f"未获取到 {self.inst_id} 在指定时间范围内的数据")
            return None
            
        return df
    
    @abstractmethod
    def preprocess_data(self, df):
        """
        预处理数据
        
        参数:
            df: 原始数据
            
        返回:
            pandas.DataFrame: 预处理后的数据
        """
        pass
    
    @abstractmethod
    def generate_signals(self, df):
        """
        生成交易信号
        
        参数:
            df: 预处理后的数据
            
        返回:
            pandas.DataFrame: 带有交易信号的数据
        """
        pass
    
    @abstractmethod
    def backtest(self, df, initial_capital=10000.0):
        """
        回测交易策略
        
        参数:
            df: 带有交易信号的数据
            initial_capital: 初始资金
            
        返回:
            dict: 回测结果
        """
        pass
    
    def run(self, start_time, end_time, initial_capital=10000.0):
        """
        运行交易模型
        
        参数:
            start_time: 开始时间
            end_time: 结束时间
            initial_capital: 初始资金
            
        返回:
            dict: 回测结果
        """
        # 获取数据
        df = self.get_data(start_time, end_time)
        if df is None:
            return None
            
        # 预处理数据
        df = self.preprocess_data(df)
        
        # 生成交易信号
        df = self.generate_signals(df)
        
        # 回测
        results = self.backtest(df, initial_capital)
        
        return results