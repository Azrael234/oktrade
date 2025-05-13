import torch
from torch.utils.data import Dataset
import numpy as np
from datetime import datetime

class KLineDataset(Dataset):
    def __init__(self, db_manager, inst_id, lookback=60, forecast=5, split_ratio=(0.8, 0.1, 0.1)):
        """
        初始化数据集
        :param db_manager: 数据库管理器
        :param inst_id: 交易对ID
        :param lookback: 回看窗口大小（分钟）
        :param forecast: 预测窗口大小（分钟）
        """
        self.inst_id = inst_id
        self.lookback = lookback
        self.forecast = forecast
        
        # 从数据库加载K线数据
        conn = db_manager.connect()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, open, high, low, close, volume 
            FROM kline_1m 
            WHERE inst_id = %s 
            ORDER BY timestamp
        """, (inst_id,))
        self.data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # 预处理数据
        self.features, self.targets = self._preprocess_data()
        
        # 数据集划分
        self.train_set, self.val_set, self.test_set = self._split_dataset(split_ratio)
    
    def _split_dataset(self, split_ratio):
        total_size = len(self.features)
        train_size = int(split_ratio[0] * total_size)
        val_size = int(split_ratio[1] * total_size)
        test_size = total_size - train_size - val_size
        
        # 按时间顺序划分数据集
        train_indices = range(0, train_size)
        val_indices = range(train_size, train_size + val_size)
        test_indices = range(train_size + val_size, total_size)
        
        return (
            (self.features[train_indices], self.targets[train_indices]),
            (self.features[val_indices], self.targets[val_indices]),
            (self.features[test_indices], self.targets[test_indices])
        )
    
    def get_train_data(self):
        return self.train_set
    
    def get_val_data(self):
        return self.val_set
    
    def get_test_data(self):
        return self.test_set
    
    def _preprocess_data(self):
        # 将数据转换为numpy数组
        prices = np.array([(x[1] + x[2] + x[3] + x[4])/4 for x in self.data])  # 使用平均价格
        volumes = np.array([x[5] for x in self.data])
        
        # 归一化
        price_mean, price_std = prices.mean(), prices.std()
        vol_mean, vol_std = volumes.mean(), volumes.std()
        
        prices = (prices - price_mean) / price_std
        volumes = (volumes - vol_mean) / vol_std
        
        # 创建特征和目标
        X, y = [], []
        for i in range(len(prices) - self.lookback - self.forecast):
            X.append(np.column_stack((prices[i:i+self.lookback], 
                                    volumes[i:i+self.lookback])))
            y.append(prices[i+self.lookback:i+self.lookback+self.forecast])
        
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]