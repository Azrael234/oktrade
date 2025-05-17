import torch
from .model import TransformerPredictor

def predict(model_path, input_data, config):
    # 加载模型
    model = TransformerPredictor(
        input_dim=2,
        model_dim=config['model_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        output_dim=1
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(config['device'])
    model.eval()
    
    # 预测
    with torch.no_grad():
        input_data = input_data.to(config['device'])
        prediction = model(input_data)
    
    return prediction.cpu().numpy()