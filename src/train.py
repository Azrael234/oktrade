import os
import logging
import os
import torch
import torch.nn as nn
from models import *
from configs.config import config_manager
from configs.data_config import data_config_manager
from typing import Literal
from pathlib import Path
from torch.utils.data import DataLoader
from data import KLineDataset
import logging
def train_model(
    db_manager, 
    inst_id: str,
    model_type: Literal['transformer', 'tcn', 'lstm', 'informer', 'autoformer'],
    mode: Literal['regression', 'classification'] = 'regression',
    experiment_name: str = 'default',
    dataset_name: str = "default",
    log_filename: str = None
):
    """统一训练接口
    
    参数:
        model_type: 模型类型
        mode: 回归或分类任务
        experiment_name: 实验名称(用于保存结果)
    """
    config = config_manager.get_config(experiment_name)
    logging.info(f"训练配置: {config}")
   
    # 1. 数据加载
    data_config = data_config_manager.get_config()
    logging.info(f"数据配置: {data_config}")
    dataset = KLineDataset(db_manager, inst_id, data_config, mode=mode)
    train_loader, val_loader = create_dataloaders(dataset, config['batch_size'])
    
    num_classes = 3
    if mode == 'classification':
        num_classes = config['num_classes']
    # 2. 模型初始化
    model = init_model(
        model_type=model_type,
        input_dim=config['input_dim'],
        output_dim=config['output_dim'] if mode == 'regression' else num_classes,  # 分类任务输出3类
        config=config,
        mode=mode
    )
    
    # 3. 训练配置
    # criterion = nn.MSELoss() if mode == 'regression' else nn.CrossEntropyLoss()
    if config.get('mode', 'regression') == 'regression':
        criterion = nn.MSELoss()
    else:
        class_weights = config['classification'].get('class_weights')
        if class_weights:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
        else:
            criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    model_name = log_filename.split('.')[0] if log_filename else f"train_{inst_id}_{model_type}_{mode}"
    # 4. 训练循环
    best_model = _train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        model_name=model_name,
        experiment_name=experiment_name
    )
    
    return best_model

def create_dataloaders(dataset, batch_size):
    """创建统一的数据加载器"""
    train_data = dataset.get_train_data()
    val_data = dataset.get_val_data()
    
    def collate_fn(batch):
        x = torch.stack([item[0] for item in batch], 0)
        y = torch.stack([item[1] for item in batch], 0)
        # 分类任务需要调整标签形状
        if len(y.shape) == 3:  # 分类任务 (batch, seq, 1) -> (batch, seq)
            y = y.squeeze(-1).long()
        return x, y
    
    train_loader = DataLoader(
        list(zip(train_data[0], train_data[1])),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        list(zip(val_data[0], val_data[1])),
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def init_model(model_type, input_dim, output_dim, config, mode):
    """统一模型初始化"""
    model_args = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'mode': mode,
        **config.get('model_args', {})
    }

     # 如果是分类任务，添加num_classes参数
    if mode == 'classification':
        model_args['num_classes'] = config['num_classes']
    
    models = {
        'transformer': TransformerPredictor,
        'tcn': TCNPredictor,
        'lstm': LSTMPredictor,
        'informer': InformerPredictor,
        # 'autoformer': AutoformerPredictor
    }
    
    return models[model_type](**model_args)

# def _train_loop(model, train_loader, val_loader, criterion, optimizer, config, model_name, experiment_name):
#     """核心训练逻辑"""
#     model = model.to(config['device'])
#     best_val_loss = float('inf')
    
#     for epoch in range(config['epochs']):
#         # 训练阶段
#         model.train()
#         train_loss = 0
#         for data, target in train_loader:
#             data, target = data.to(config['device']), target.to(config['device'])
            
#             optimizer.zero_grad()
#             output = model(data)
            
#             # 分类任务需要调整target形状
#             if isinstance(criterion, nn.CrossEntropyLoss):
#                 target = target[:, -1]  # 只取最后一个时间步的标签
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
        
#         # 验证阶段
#         val_loss = evaluate(model, val_loader, criterion, config['device'])
        
#         # 打印日志
#         # print(f"Epoch {epoch+1}/{config['epochs']} | "
#         #       f"Train Loss: {train_loss/len(train_loader):.4f} | "
#         #       f"Val Loss: {val_loss/len(val_loader):.4f}")
#         logging.info(f"Epoch {epoch+1}/{config['epochs']} | "
#                      f"Train Loss: {train_loss/len(train_loader):.4f} | "
#                      f"Val Loss: {val_loss/len(val_loader):.4f}")
                
        
#         # 保存最佳模型
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             save_model(model, experiment_name, model_name)
    
#     return model

def _train_loop(model, train_loader, val_loader, criterion, optimizer, config, model_name, experiment_name):
    """核心训练逻辑"""
    model = model.to(config['device'])
    best_val_loss = float('inf')
    
    # 配置参数
    eval_interval = config['eval_interval']  # 评估间隔
    save_interval = config['save_interval']  # 模型保存间隔
    
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(config['device']), target.to(config['device'])
            
            optimizer.zero_grad()
            output = model(data)
            
            if isinstance(criterion, nn.CrossEntropyLoss):
                target = target[:, -1]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 评估和保存逻辑
        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == config['epochs']:
            val_loss = evaluate(model, val_loader, criterion, config['device'])
            logging.info(f"Epoch {epoch+1}/{config['epochs']} | "
                        f"Train Loss: {train_loss/len(train_loader):.4f} | "
                        f"Val Loss: {val_loss/len(val_loader):.4f}")
            
            # 更新最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, experiment_name, model_name+"_best")
                logging.info(f"New best model saved at epoch {epoch+1}")
        
        
        
        # 按间隔保存模型
        if (epoch + 1) % save_interval == 0:
            save_model(model, experiment_name,  model_name+f"_epoch_{epoch+1}")
            # 保存最新模型（始终覆盖）
            save_model(model, experiment_name,  model_name+"_latest")
    
    return model

def evaluate(model, loader, criterion, device):
    """评估函数"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if isinstance(criterion, nn.CrossEntropyLoss):
                target = target[:, -1]  # 分类任务调整
                
            total_loss += criterion(output, target).item()
    return total_loss


def save_model(model, experiment_name, save_name, root_dir="logs"):
    """改进的模型保存函数"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), root_dir)
    save_dir = Path(log_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 统一保存为.pth文件
    save_path = save_dir / f"{save_name}.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config if hasattr(model, 'config') else None
    }, save_path)
    
    # logging.info(f"Model '{save_name}' saved to: {save_path}")