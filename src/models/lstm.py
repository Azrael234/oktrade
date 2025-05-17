import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, src):
        # src: (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, src.size(0), self.hidden_dim).to(src.device)
        c0 = torch.zeros(self.num_layers, src.size(0), self.hidden_dim).to(src.device)
        
        # LSTM前向传播
        out, _ = self.lstm(src, (h0, c0))
        
        # 只取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out