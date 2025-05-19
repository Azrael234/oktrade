import torch
import torch.nn as nn
import math

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=2, model_dim=64, num_heads=4, 
                 num_layers=3, output_dim=5, dropout=0.1, mode='regression', num_classes=None):
        """
        参数:
        - mode: 'regression' 或 'classification'
        - num_classes: 分类任务的类别数（仅在 mode='classification' 时生效）
        - output_dim: 回归任务预测的时间步数（仅在 mode='regression' 时生效）
        """
        super().__init__()
        self.mode = mode
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.output_dim = output_dim
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        
        # Transformer 编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads,
            dim_feedforward=4*model_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        if mode == 'regression':
            # 回归任务：预测 output_dim 个时间步的值（每个时间步1个值）
            self.output_proj = nn.Linear(model_dim, 1)  # 输出 shape: [batch, output_dim]
        elif mode == 'classification':
            # 分类任务：预测最后一个时间步的类别概率
            assert num_classes is not None, "num_classes must be specified for classification mode"
            self.output_proj = nn.Sequential(
                nn.Linear(model_dim, model_dim // 2),
                nn.ReLU(),
                nn.Linear(model_dim // 2, num_classes),
                nn.Softmax(dim=-1)  # 输出 shape: [batch, num_classes]
            )
        else:
            raise ValueError("mode must be 'regression' or 'classification'")

    def forward(self, src):
        # src: (batch, seq_len, input_dim)
        src = self.input_proj(src)  # (batch, seq_len, model_dim)
        src = self.pos_encoder(src)
        
        # Transformer 编码
        output = self.transformer(src)  # (batch, seq_len, model_dim)
        
        # 根据任务类型处理输出
        if self.mode == 'regression':
            # 取最后 output_dim 个时间步，每个时间步预测1个值
            output = output[:, -self.output_dim:, :]  # (batch, output_dim, model_dim)
            output = self.output_proj(output)  # (batch, output_dim, 1)
            output = output.squeeze(-1)  # (batch, output_dim)
        elif self.mode == 'classification':
            # 取最后一个时间步，预测类别概率
            output = output[:, -1, :]  # (batch, model_dim)
            output = self.output_proj(output)  # (batch, num_classes)
        
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)