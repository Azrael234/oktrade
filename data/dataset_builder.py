import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json
from database.db_schema import DatabaseManager
from data.aggregator import KlineAggregator

class KlineDatasetBuilder:
    def __init__(self, db_manager):
        """
        K线数据集构建器
        
        参数:
            db_manager (DatabaseManager): 数据库管理器
        """
        self.db_manager = db_manager
        self.aggregator = KlineAggregator(db_manager)
        self.logger = logging.getLogger(__name__)
        
        # 涨跌幅档次
        self.price_change_levels = [0.03, 0.05, 0.08, 0.10]  # 3%, 5%, 8%, 10%
        
    def get_aligned_kline_data(self, main_symbol, start_time, end_time, window_size=14400):
        """
        获取时间对齐的K线数据
        
        参数:
            main_symbol (str): 主交易对 (DOT-USDT 或 BNB-USDT)
            start_time (datetime): 开始时间
            end_time (datetime): 结束时间
            window_size (int): 窗口大小（分钟数，默认14400=10天）
            
        返回:
            pandas.DataFrame: 对齐后的数据
        """
        # 获取各个交易对的1分钟K线数据
        main_df = self.aggregator.get_1m_kline(main_symbol, start_time, end_time)
        btc_df = self.aggregator.get_1m_kline('BTC-USDT', start_time, end_time)
        eth_df = self.aggregator.get_1m_kline('ETH-USDT', start_time, end_time)
        
        if main_df.empty or btc_df.empty or eth_df.empty:
            self.logger.error(f"缺少必要的K线数据: {main_symbol}, BTC-USDT, ETH-USDT")
            return pd.DataFrame()
        
        # 时间对齐 - 使用内连接确保所有时间点都有数据
        aligned_df = main_df.join(btc_df, how='inner', rsuffix='_btc')
        aligned_df = aligned_df.join(eth_df, how='inner', rsuffix='_eth')
        
        # 重命名列
        main_prefix = main_symbol.split('-')[0].lower()
        columns_mapping = {
            'open': f'{main_prefix}_open',
            'high': f'{main_prefix}_high', 
            'low': f'{main_prefix}_low',
            'close': f'{main_prefix}_close',
            'volume': f'{main_prefix}_volume',
            'open_btc': 'btc_open',
            'high_btc': 'btc_high',
            'low_btc': 'btc_low', 
            'close_btc': 'btc_close',
            'volume_btc': 'btc_volume',
            'open_eth': 'eth_open',
            'high_eth': 'eth_high',
            'low_eth': 'eth_low',
            'close_eth': 'eth_close',
            'volume_eth': 'eth_volume'
        }
        aligned_df = aligned_df.rename(columns=columns_mapping)
        
        # 处理缺失数据 - 前向填充
        aligned_df = aligned_df.fillna(method='ffill')
        
        # 检查连续缺失数据并标记
        for col in aligned_df.columns:
            # 如果连续缺失>=3个点，标记为无效
            mask = aligned_df[col].isna()
            consecutive_na = mask.groupby((~mask).cumsum()).cumsum()
            aligned_df.loc[consecutive_na >= 3, 'invalid_sample'] = True
        
        return aligned_df
    
    def calculate_future_returns(self, df, main_symbol, future_minutes=10):
        """
        计算未来涨跌幅
        
        参数:
            df (pandas.DataFrame): K线数据
            main_symbol (str): 主交易对
            future_minutes (int): 未来分钟数
            
        返回:
            pandas.DataFrame: 添加了未来收益率的数据
        """
        main_prefix = main_symbol.split('-')[0].lower()
        close_col = f'{main_prefix}_close'
        high_col = f'{main_prefix}_high'
        low_col = f'{main_prefix}_low'
        
        # 计算未来10分钟的最高价和最低价
        df[f'future_{future_minutes}m_high'] = df[high_col].rolling(
            window=future_minutes, min_periods=1
        ).max().shift(-future_minutes)
        
        df[f'future_{future_minutes}m_low'] = df[low_col].rolling(
            window=future_minutes, min_periods=1
        ).min().shift(-future_minutes)
        
        # 计算当前价格
        current_price = df[close_col]
        
        # 计算未来最大涨幅和最大跌幅
        max_up_return = (df[f'future_{future_minutes}m_high'] - current_price) / current_price
        max_down_return = (df[f'future_{future_minutes}m_low'] - current_price) / current_price
        
        # 为每个涨跌幅档次创建标签
        for level in self.price_change_levels:
            # 上涨概率标签
            df[f'can_rise_{int(level*100)}pct'] = (max_up_return >= level).astype(int)
            # 下跌概率标签  
            df[f'can_fall_{int(level*100)}pct'] = (max_down_return <= -level).astype(int)
        
        # 添加最大涨跌幅
        df['max_up_return_10m'] = max_up_return
        df['max_down_return_10m'] = max_down_return
        
        return df
    
    def build_training_dataset(self, main_symbol, start_date, end_date, window_size=14400):
        """
        构建训练数据集
        
        参数:
            main_symbol (str): 主交易对 (DOT-USDT 或 BNB-USDT)
            start_date (datetime): 开始日期
            end_date (datetime): 结束日期
            window_size (int): 滑动窗口大小（分钟）
            
        返回:
            pandas.DataFrame: 训练数据集
        """
        self.logger.info(f"开始构建 {main_symbol} 的训练数据集")
        
        # 获取对齐的K线数据
        aligned_df = self.get_aligned_kline_data(main_symbol, start_date, end_date)
        
        if aligned_df.empty:
            self.logger.error("无法获取对齐的K线数据")
            return pd.DataFrame()
        
        # 计算未来收益率
        dataset = self.calculate_future_returns(aligned_df, main_symbol)
        
        # 移除无效样本
        if 'invalid_sample' in dataset.columns:
            dataset = dataset[dataset['invalid_sample'] != True]
            dataset = dataset.drop('invalid_sample', axis=1)
        
        # 移除最后10分钟的数据（因为没有未来数据）
        dataset = dataset.iloc[:-10]
        
        self.logger.info(f"成功构建训练数据集，共 {len(dataset)} 条记录")
        return dataset
    
    def save_dataset_to_parquet(self, dataset, main_symbol, date_str):
        """
        保存数据集为Parquet格式
        
        参数:
            dataset (pandas.DataFrame): 数据集
            main_symbol (str): 主交易对
            date_str (str): 日期字符串 (YYMMDD格式)
        """
        if dataset.empty:
            self.logger.warning("数据集为空，无法保存")
            return
        
        # 分离主币数据和BTC/ETH数据
        main_prefix = main_symbol.split('-')[0].lower()
        
        # 主币数据
        main_cols = [col for col in dataset.columns if col.startswith(main_prefix) or 
                    col.startswith('can_') or col.startswith('max_') or col.startswith('future_')]
        main_cols.insert(0, 'timestamp') if 'timestamp' not in main_cols else None
        
        main_dataset = dataset[main_cols].copy()
        main_filename = f"data/{date_str}_{main_prefix.upper()}.parquet"
        main_dataset.to_parquet(main_filename)
        self.logger.info(f"主币数据集已保存: {main_filename}")
        
        # BTC/ETH对齐数据
        btc_eth_cols = [col for col in dataset.columns if col.startswith('btc_') or col.startswith('eth_')]
        btc_eth_cols.insert(0, 'timestamp') if 'timestamp' not in btc_eth_cols else None
        
        btc_eth_dataset = dataset[btc_eth_cols].copy()
        btc_eth_filename = f"data/{date_str}_BTC_ETH_aligned.parquet"
        btc_eth_dataset.to_parquet(btc_eth_filename)
        self.logger.info(f"BTC/ETH对齐数据集已保存: {btc_eth_filename}")
        
        return main_filename, btc_eth_filename

# 加载配置文件函数
def load_config():
    """
    加载配置文件
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"加载配置文件失败: {str(e)}")
        return {}

# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 加载配置文件
    config = load_config()
    db_config = config.get('database', {})
    
    # 初始化数据库管理器
    db_manager = DatabaseManager(
        host=db_config.get('host', 'localhost'),
        database=db_config.get('database', 'oktrade'), 
        user=db_config.get('user', 'postgres'),
        password=db_config.get('password', 'your_password'),
        port=db_config.get('port', 5432)
    )
    
    # 创建数据集构建器
    builder = KlineDatasetBuilder(db_manager)
    
    # 设置时间范围（最近10天）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    
    # 构建DOT数据集
    dot_dataset = builder.build_training_dataset('DOT-USDT', start_date, end_date)
    if not dot_dataset.empty:
        date_str = end_date.strftime('%y%m%d')
        builder.save_dataset_to_parquet(dot_dataset, 'DOT-USDT', date_str)
    
    # 构建BNB数据集
    bnb_dataset = builder.build_training_dataset('BNB-USDT', start_date, end_date)
    if not bnb_dataset.empty:
        date_str = end_date.strftime('%y%m%d')
        builder.save_dataset_to_parquet(bnb_dataset, 'BNB-USDT', date_str)