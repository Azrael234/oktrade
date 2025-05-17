import torch
from torch.utils.data import DataLoader
from .data import KLineDataset
from .models import TCNPredictor, LSTMPredictor, TransformerPredictor
import os
from configs.config import config_manager

def train_transformer(db_manager, inst_id, experiment_name='default'):
    # 获取配置
    config = config_manager.get_config()
    # 初始化数据集
    dataset = KLineDataset(db_manager, inst_id)
    train_data = dataset.get_train_data()
    val_data = dataset.get_val_data()
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'])
    
    # 初始化Transformer模型
    model = TransformerPredictor(
        input_dim=2,
        model_dim=config['model_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        output_dim=5  # 预测未来5分钟的价格
    )
    return _train_model(model, train_loader, val_loader, config, f"{inst_id}_transformer.pth")

def train_tcn(db_manager, inst_id, experiment_name='default'):
    # 获取配置
    config = config_manager.get_config()
    # 初始化数据集
    dataset = KLineDataset(db_manager, inst_id)
    train_data = dataset.get_train_data()
    val_data = dataset.get_val_data()
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'])
    
    # 初始化TCN模型
    model = TCNPredictor(
        input_dim=2,
        output_dim=5,
        num_channels=[64, 64, 64],
        kernel_size=3
    )
    return _train_model(model, train_loader, val_loader, config, f"{inst_id}_tcn.pth")

def train_lstm(db_manager, inst_id, experiment_name='default'):
    # 获取配置
    config = config_manager.get_config()
    # 初始化数据集
    dataset = KLineDataset(db_manager, inst_id)
    train_data = dataset.get_train_data()
    val_data = dataset.get_val_data()
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'])
    
    # 初始化LSTM模型
    model = LSTMPredictor(
        input_dim=2,
        hidden_dim=64,
        num_layers=2,
        output_dim=5
    )
    return _train_model(model, train_loader, val_loader, config, f"{inst_id}_lstm.pth")

def _train_model(model, train_loader, val_loader, config, model_name):
    model = model.to(config['device'])
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # 训练循环
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config['device']), target.to(config['device'])
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证集评估
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(config['device']), target.to(config['device'])
                output = model(data)
                val_loss += criterion(output, target).item()
        
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    # 保存模型
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f"models/{model_name}")
    return model