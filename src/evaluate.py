import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data import KLineDataset
from models import *
from typing import Literal, Dict, List
import os
import logging
from sklearn.metrics import mean_absolute_error, confusion_matrix
import seaborn as sns
from configs.config import config_manager
from configs.data_config import data_config_manager
from utils.visualize import plot_comparison, visualize_results

def evaluate_model(db_manager, model_path: str, experiment_name: str, dataset_name: str = "default", mode: Literal['regression', 'classification'] = 'regression'):
    """统一评估函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 1. 加载模型
        config = config_manager.get_config(experiment_name)
        logging.info(f"加载配置: {config}")

        model = load_model_from_path(model_path, config, mode, device)
        logging.info(f"成功加载模型: {model_path}")
        
        # 2. 准备数据
        data_config = data_config_manager.get_config(dataset_name)
        logging.info(f"加载数据集配置: {data_config}")
        dataset = KLineDataset(db_manager, model.inst_id, data_config=data_config, mode=mode)
        test_loader = create_dataloaders(dataset, batch_size=config['batch_size'], mode='test')
        
        # 3. 评估
        criterion = nn.MSELoss() if mode == 'regression' else nn.CrossEntropyLoss()
        test_loss, predictions, targets = _evaluate(model, test_loader, criterion, device, mode)
        
        # 4. 计算指标
        metrics = calculate_metrics(predictions, targets, mode)
        
        # 5. 可视化结果
        visualize_name = model_path.split(os.path.sep)[-2]  # 从路径中提取模型名称
        visualize_results(predictions, targets, mode, data_config['thresholds'], model_name=visualize_name)
        
        return {
            'test_loss': test_loss,
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets
        }
    except Exception as e:
        logging.error(f"评估模型失败: {str(e)}")
        raise

def compare_models(db_manager, experiment: str, dataset: str, model_paths: List[str], inst_id: str, mode: str, config: Dict):
    """比较多个模型性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    try:
        for path in model_paths:
            model_name = os.path.basename(path).split('.')[0]
            logging.info(f"开始评估模型: {model_name}")
            
            # 使用统一的评估函数
            experiment_name = model_path.split(os.path.sep)[-2]
            results[model_name] = evaluate_model(
                db_manager=db_manager,
                model_path=path,
                experiment_name=experiment_name,
                dataset_name=dataset,
                mode=mode
            )
        
        # 可视化比较
        plot_comparison(results, mode)
        return results
        
    except Exception as e:
        logging.error(f"比较模型失败: {str(e)}")
        raise

def _evaluate(model, loader, criterion, device, mode: str):
    """核心评估逻辑"""
    model.eval()
    total_loss = 0
    preds, targets = [], []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if mode == 'classification':
                target = target.squeeze(-1)  # 分类任务调整标签
            
            loss = criterion(output, target)
            total_loss += loss.item()

            preds.append(output.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
    
    return (total_loss / len(loader), 
            np.concatenate(preds), 
            np.concatenate(targets))

def load_model_from_path(path: str, config, mode, device):
    """从路径加载模型"""
    try:
        # 从文件名解析模型信息
        filename = os.path.basename(path)
        parts = filename.split('_')
        
        if len(parts) < 5:
            raise ValueError("模型文件名格式不正确，应为: train_<inst>_<model_type>_<mode>_<model_stage>.pth")
            
        inst_id = parts[1]
        model_type = parts[2]
        mode = parts[3]

        model_args = {
            'input_dim': config["input_dim"],
            'output_dim': config["output_dim"],
            'mode': mode,
            **config.get('model_args', {})
        }
        
        if mode == 'classification':
            model_args['num_classes'] = config["num_classes"]
        # 根据模型类型初始化
        model_classes = {
            'transformer': TransformerPredictor,
            'tcn': TCNPredictor,
            'lstm': LSTMPredictor,
            'informer': InformerPredictor
            # 'autoformer': AutoformerPredictor
        }
        
        if model_type not in model_classes:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        model = model_classes[model_type](**model_args).to(device)
        model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])
        model.inst_id = inst_id  # 保存交易对信息
        return model
        
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        raise

def create_dataloaders(dataset, batch_size: int, mode: str = 'train'):
    """创建数据加载器"""
    data = dataset.get_train_data() if mode == 'train' else dataset.get_test_data()
    
    def collate_fn(batch):
        x = torch.stack([item[0] for item in batch], 0)
        y = torch.stack([item[1] for item in batch], 0)
        return x, y
    
    return DataLoader(
        list(zip(data[0], data[1])),
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        collate_fn=collate_fn
    )

def calculate_metrics(preds, targets, mode: str):
    """计算评估指标"""
    metrics = {}
    
    if mode == 'regression':
        metrics['mae'] = mean_absolute_error(targets, preds)
        metrics['rmse'] = np.sqrt(np.mean((targets - preds)**2))
    else:
        pred_classes = np.argmax(preds, axis=1)
        target_classes = np.argmax(targets, axis=1)
        print(f"pred_classes: {pred_classes}")
        print(f"target_classes: {target_classes}")
        metrics['accuracy'] = np.mean(pred_classes == target_classes)
        
    return metrics

# def visualize_results(preds, targets, mode: str, model_name: str = None):
#     """可视化结果"""
#     plt.figure(figsize=(12, 6))
#     title = f'{mode.capitalize()} Results'
#     if model_name:
#         title += f' - {model_name}'
    
#     if mode == 'regression':
#         plt.plot(preds[:200, 0], label='Prediction', alpha=0.8)
#         plt.plot(targets[:200, 0], label='True', alpha=0.5)
#         plt.ylabel('Price')
#     else:
#         pred_classes = np.argmax(preds, axis=1)
#         target_classes = np.argmax(targets, axis=1)
#         cm = confusion_matrix(target_classes, pred_classes)
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                    xticklabels=['Down', 'Flat', 'Up'],
#                    yticklabels=['Down', 'Flat', 'Up'])
#         plt.ylabel('True')
#         plt.xlabel('Predicted')
    
#     plt.title(title)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

