import logging
import pandas as pd
import numpy as np
from database.db_schema import DatabaseManager

class KlineAggregator:
    def __init__(self, db_manager):
        """
        K线数据合成器
        
        参数:
            db_manager (DatabaseManager): 数据库管理器
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    def get_1m_kline(self, inst_id, start_time, end_time):
        """
        获取1分钟K线数据
        
        参数:
            inst_id (str): 交易对ID
            start_time (datetime): 开始时间
            end_time (datetime): 结束时间
            
        返回:
            pandas.DataFrame: 1分钟K线数据
        """
        conn = self.db_manager.connect()
        
        try:
            query = """
            SELECT timestamp, open, high, low, close, volume
            FROM kline_1m
            WHERE inst_id = %s AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, conn, params=(inst_id, start_time, end_time))
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"获取1分钟K线数据失败: {str(e)}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def aggregate_kline(self, df_1m, period):
        """
        合成其他周期的K线数据
        
        参数:
            df_1m (pandas.DataFrame): 1分钟K线数据
            period (str): 目标周期，如'5min', '15min', '1h', '4h', '1d'
            
        返回:
            pandas.DataFrame: 合成后的K线数据
        """
        if df_1m.empty:
            return pd.DataFrame()
            
        # 确保数据按时间排序
        df_1m = df_1m.sort_index()
        
        # 根据周期重采样
        resampled = df_1m.resample(period)
        
        # 合成OHLCV数据
        df_period = pd.DataFrame({
            'open': resampled['open'].first(),
            'high': resampled['high'].max(),
            'low': resampled['low'].min(),
            'close': resampled['close'].last(),
            'volume': resampled['volume'].sum()
        })
        
        return df_period
    
    def get_aggregated_kline(self, inst_id, start_time, end_time, period):
        """
        获取合成后的K线数据
        
        参数:
            inst_id (str): 交易对ID
            start_time (datetime): 开始时间
            end_time (datetime): 结束时间
            period (str): 目标周期，如'5min', '15min', '1h', '4h', '1d'
            
        返回:
            pandas.DataFrame: 合成后的K线数据
        """
        # 获取1分钟K线数据
        df_1m = self.get_1m_kline(inst_id, start_time, end_time)
        
        if df_1m.empty:
            self.logger.warning(f"未找到 {inst_id} 在指定时间范围内的1分钟K线数据")
            return pd.DataFrame()
        
        # 合成目标周期的K线数据
        df_period = self.aggregate_kline(df_1m, period)
        
        return df_period