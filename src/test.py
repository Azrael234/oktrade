import torch
from torch.utils.data import DataLoader, random_split
from .data_loader import KLineDataset
from .model import TCNPredictor
from .lstm import LSTMPredictor
from .transformer import TransformerPredictor
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions, targets = [], []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
    
    return total_loss / len(test_loader), np.concatenate(predictions), np.concatenate(targets)

def compare_models(db_manager, inst_id, config):
    # 加载数据集
    dataset = KLineDataset(db_manager, inst_id)
    test_data = dataset.get_test_data()
    test_loader = DataLoader(test_data, batch_size=config['batch_size'])
    
    # 初始化模型
    models = {
        'TCN': TCNPredictor(
            input_dim=2,
            output_dim=5,
            num_channels=[64, 64, 64],
            kernel_size=3
        ),
        'LSTM': LSTMPredictor(
            input_dim=2,
            hidden_dim=64,
            num_layers=2,
            output_dim=5
        ),
        'Transformer': TransformerPredictor(
            input_dim=2,
            model_dim=64,
            num_heads=4,
            num_layers=2,
            output_dim=5
        )
    }
    
    # 训练和评估
    results = {}
    criterion = torch.nn.MSELoss()
    
    for name, model in models.items():
        print(f"Training {name} model...")
        model = model.to(config['device'])
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        
        # 训练
        for epoch in range(config['epochs']):
            model.train()
            for data, target in train_loader:
                data, target = data.to(config['device']), target.to(config['device'])
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # 评估
        test_loss, predictions, targets = evaluate_model(model, test_loader, criterion, config['device'])
        results[name] = {
            'test_loss': test_loss,
            'predictions': predictions,
            'targets': targets
        }
        print(f"{name} Test Loss: {test_loss:.4f}")
    
    # 可视化结果
    plt.figure(figsize=(12, 6))
    for name, result in results.items():
        plt.plot(result['predictions'][:, 0], label=f'{name} Prediction')
    plt.plot(results['TCN']['targets'][:, 0], label='True Value', alpha=0.5)
    plt.legend()
    plt.title('Model Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.show()
    
    return results