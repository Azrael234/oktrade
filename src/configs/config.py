import json
import os

class ExperimentConfig:
    def __init__(self, batch_size=32, model_dim=128, num_heads=8, num_layers=4, lr=0.0001, epochs=50):
        self.batch_size = batch_size
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def to_dict(self):
        return {
            'batch_size': self.batch_size,
            'model_dim': self.model_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'lr': self.lr,
            'epochs': self.epochs,
            'device': self.device
        }

    def save(self, experiment_name):
        """保存当前配置到文件"""
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments')
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, f'{experiment_name}.json')
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, experiment_name):
        """从文件加载配置"""
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments')
        config_path = os.path.join(config_dir, f'{experiment_name}.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(**config)

# 添加默认实验配置
config_manager.add_experiment('default', {
    'batch_size': 32,
    'model_dim': 128,
    'num_heads': 8,
    'num_layers': 4,
    'lr': 0.0001,
    'epochs': 50
})

# 设置默认实验为当前活动实验
config_manager.set_active_experiment('default')