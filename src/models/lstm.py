import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, 
                 output_dim=5, dropout=0.1, mode='regression', num_classes=None):
        super().__init__()
        self.mode = mode
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        if mode == 'regression':
            self.output_layer = nn.Linear(hidden_dim, 1)  # 每个时间步预测1个值
        else:
            assert num_classes is not None, "分类任务必须指定 num_classes"
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, num_classes),
                nn.Softmax(dim=-1) if num_classes > 1 else nn.Sigmoid()
            )

    def forward(self, src):
        out, _ = self.lstm(src)  # (batch, seq_len, hidden_dim)
        
        if self.mode == 'regression':
            # 取最后 output_dim 个时间步，每个预测1个值
            output = out[:, -self.output_dim:, :]  # (batch, output_dim, hidden_dim)
            output = self.output_layer(output)  # (batch, output_dim, 1)
            output = output.squeeze(-1)  # (batch, output_dim)
        else:
            # 分类任务：取最后一个时间步
            output = out[:, -1, :]  # (batch, hidden_dim)
            output = self.output_layer(output)  # (batch, num_classes)
        
        return output