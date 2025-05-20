import json
import os
import torch
from pathlib import Path

class DataConfigManager:
    def __init__(self, config_dir="data_configs"):
        self.configs = {}
        self.active_config = None
        self.config_dir = config_dir
        self.configs = {}
        self.active_experiment = None
        self.config_dir = config_dir
        # 确保配置目录存在
        # 获取当前脚本所在目录的绝对路径
        script_dir = Path(__file__).parent.absolute()
        self.config_dir = Path(os.path.join(script_dir, config_dir))  # 转为 Path
        self.config_dir.mkdir(parents=True, exist_ok=True)
        # 加载所有现有配置
        self._load_all_configs()

    def _load_all_configs(self):
        """加载experiments目录下所有的JSON配置文件"""
        for config_file in Path(self.config_dir).glob("*.json"):
            experiment_name = config_file.stem
            with open(config_file, 'r') as f:
                self.configs[experiment_name] = json.load(f)
                # 自动设置device

    def add_experiment(self, name, config):
        """添加新实验配置并保存到JSON文件"""
        self.configs[name] = config
        self._save_config(name)
        
    def _save_config(self, name):
        """将配置保存到JSON文件"""
        config_path = Path(self.config_dir) / f"{name}.json"
        with open(config_path, 'w') as f:
            json.dump(self.configs[name], f, indent=4)

    def get_config(self, name='default'):
        """获取指定实验配置，如果不存在则返回空字典"""
        return self.configs.get(name, {})

    def set_active_experiment(self, name):
        """设置当前活动实验"""
        if name in self.configs:
            self.active_experiment = name
        else:
            raise ValueError(f"Experiment '{name}' not found in configurations")

# 初始化配置管理器
data_config_manager = DataConfigManager()

# 如果没有任何配置，添加默认配置
if not data_config_manager.configs:
    default_config = {
            "database": "oktrade",
            "lookback": 60,
            "forecast": 5,
            "split_ratio": [0.8, 0.1, 0.1],
            "use_label_smoothing": True,
            "thresholds": [-0.5, -0.2, 0, 0.2, 0.5],
            "class_weights": None
    }
    data_config_manager.add_experiment('default', default_config)

# 设置默认实验为当前活动实验
data_config_manager.set_active_experiment('default')