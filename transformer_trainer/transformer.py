import torch
import torch.nn as nn

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 60, model_dim))
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
            dropout=dropout
        )
        self.output_proj = nn.Linear(model_dim, output_dim)
    
    def forward(self, src):
        # src: (batch_size, seq_len, input_dim)
        src = self.input_proj(src) + self.positional_encoding
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, model_dim)
        tgt = torch.zeros_like(src)
        output = self.transformer(src, tgt)
        output = output.permute(1, 0, 2)  # (batch_size, seq_len, model_dim)
        return self.output_proj(output)