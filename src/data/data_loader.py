import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, List

class KLineDataset(Dataset):
    def __init__(self, db_manager, inst_id: str, 
                 lookback: int = 60, 
                 forecast: int = 5,
                 split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 mode: str = 'regression',
                 thresholds: List[float] = None):
        """
        参数:
        - mode: 'regression' 或 'classification'
        - thresholds: 分类阈值列表(单位: 百分比变化)
            例如: [-0.5, -0.2, 0, 0.2, 0.5] 表示:
                class 0: < -0.5%
                class 1: -0.5% ~ -0.2%
                class 2: -0.2% ~ 0%
                class 3: 0% ~ 0.2%
                class 4: 0.2% ~ 0.5%
                class 5: > 0.5%
        """
        self.inst_id = inst_id
        self.lookback = lookback
        self.forecast = forecast
        self.mode = mode
        self.thresholds = [-0.5, -0.2, 0, 0.2, 0.5] if thresholds is None else thresholds
        
        # 加载原始数据
        self._load_data(db_manager)
        
        # 预处理
        self.features, self.targets = self._preprocess_data()
        
        # 数据集划分
        self.train_set, self.val_set, self.test_set = self._split_dataset(split_ratio)

    def _load_data(self, db_manager):
        """从数据库加载原始数据"""
        conn = db_manager.connect()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, open, high, low, close, volume 
            FROM kline_1m 
            WHERE inst_id = %s 
            ORDER BY timestamp
        """, (self.inst_id,))
        self.raw_data = cursor.fetchall()
        cursor.close()
        conn.close()

    def _preprocess_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # 提取价格和成交量
        prices = np.array([(x[1] + x[2] + x[3] + 2 * x[4])/5 for x in self.raw_data])  # 使用平均价格,加大close的权重
        volumes = np.array([float(x[5]) for x in self.raw_data])
        
        # 标准化
        self.price_mean, self.price_std = prices.mean(), prices.std()
        self.vol_mean, self.vol_std = volumes.mean(), volumes.std()
        norm_prices = (prices - self.price_mean) / self.price_std
        norm_volumes = (volumes - self.vol_mean) / self.vol_std
        
        # 构建特征和目标
        features, targets = [], []
        for i in range(len(prices) - self.lookback - self.forecast):
            # 特征: (lookback, 2)
            feature = np.column_stack((
                norm_prices[i:i+self.lookback],
                norm_volumes[i:i+self.lookback]
            )).astype(np.float32)
            
            # 目标处理
            if self.mode == 'regression':
                target = norm_prices[i+self.lookback:i+self.lookback+self.forecast]
            else:
                # 计算实际价格变化百分比
                current_price = closes[i+self.lookback-1]
                future_prices = closes[i+self.lookback:i+self.lookback+self.forecast]
                price_changes = (future_prices - current_price) / current_price * 100
                
                # 转换为类别 (取平均变化或每个时间步单独分类)
                avg_change = np.mean(price_changes)
                target = self._change_to_class(avg_change)
                target = self._get_smoothed_labels(target)  # 标签平滑
                # 或者对每个时间步分类:
                # target = [self._change_to_class(chg) for chg in price_changes]
            
            features.append(feature)
            targets.append(target)
        
        # 转换为张量
        features_tensor = torch.stack([torch.FloatTensor(f) for f in features])
        
        if self.mode == 'regression':
            targets_tensor = torch.stack([torch.FloatTensor(t) for t in targets])
        else:
            targets_tensor = torch.LongTensor(targets)  # 分类任务用LongTensor
            
        return features_tensor, targets_tensor

    def _change_to_class(self, change: float) -> int:
        """将价格变化百分比转换为类别"""
        for i, thresh in enumerate(self.thresholds):
            if change < thresh:
                return i
        return len(self.thresholds)  # 最后一类

    def get_num_classes(self) -> int:
        """获取类别数量"""
        return len(self.thresholds) + 1

    def _get_smoothed_labels(self, class_idx):
        """标签平滑"""
        num_classes = len(self.thresholds) + 1
        smooth_val = 0.1  # 平滑系数
        labels = torch.full((num_classes,), smooth_val/(num_classes-1))
        labels[class_idx] = 1.0 - smooth_val
        return labels
    
    def _split_dataset(self, split_ratio):
        total_size = len(self.features)
        train_size = int(split_ratio[0] * total_size)
        val_size = int(split_ratio[1] * total_size)
        test_size = total_size - train_size - val_size
        
        # 按时间顺序划分数据集
        # 修改索引类型为list
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_size))
        
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
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]