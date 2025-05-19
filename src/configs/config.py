import json
import os
import torch

class ConfigManager:
    def __init__(self):
        self.configs = {}
        self.active_experiment = None

    def add_experiment(self, name, config):
        self.configs[name] = config

    def get_config(self, name='default'):
        return self.configs.get(name, {})

    def set_active_experiment(self, name):
        self.active_experiment = name

config_manager = ConfigManager()
# 添加默认实验配置
config_manager.add_experiment('default', {
    'batch_size': 32,
    'input_dim': 2,  
    'output_dim': 5,
    'model_dim': 128,
    'num_heads': 8,
    'num_layers': 4,
    'lr': 0.0001,
    'epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'classification': {
                    'thresholds': [-0.5, -0.2, 0, 0.2, 0.5],  # 默认6类
                    'class_weights': None  # 可设置类别权重
                }
})

# 设置默认实验为当前活动实验
config_manager.set_active_experiment('default')