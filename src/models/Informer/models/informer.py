import torch
import torch.nn as nn

# from .attn import FullAttention, ProbAttention, AttentionLayer
# from .encoder import Encoder, EncoderLayer, ConvLayer
# from .decoder import Decoder, DecoderLayer
# from .embed import DataEmbedding
from .model import Informer, InformerStack

class InformerPredictor(nn.Module):
    def __init__(self, input_dim=2, model_dim=512, num_heads=8, 
                 enc_layers=3, dec_layers=2, output_dim=5, dropout=0.1,
                 mode='regression', num_classes=None, factor=5, 
                 activation='gelu', attn='prob', distil=True, mix=True):
        """
        统一接口的 Informer 模型封装
        
        参数:
        - input_dim: 输入特征维度
        - model_dim: 模型维度（d_model）
        - num_heads: 注意力头数
        - enc_layers/dec_layers: 编码器/解码器层数
        - output_dim: 输出序列长度（预测长度）
        - mode: 'regression' 或 'classification'
        - num_classes: 分类任务类别数（仅当 mode == 'classification' 时有效）
        - factor: ProbSparse Attention 的采样因子
        - activation: 激活函数 ('gelu' or 'relu')
        - attn: 注意力类型 ('prob' or 'full')
        - distil: 是否使用蒸馏（下采样）
        - mix: 是否混合 Q/K/V 排列
        """
        super().__init__()
        self.mode = mode
        self.output_dim = output_dim

        # 默认 seq_len 等于 input_seq_len (假设训练时输入是固定长度)
        self.seq_len = 96   # 可以根据数据集调整
        self.label_len = 48
        self.pred_len = output_dim

        # 使用 encoder 的 input_dim 作为 decoder 调用的 dummy dim
        self.model = Informer(
            enc_in=input_dim,
            dec_in=input_dim,
            c_out=output_dim if mode == 'regression' else num_classes,
            seq_len=self.seq_len,
            label_len=self.label_len,
            out_len=self.pred_len,
            factor=factor,
            d_model=model_dim,
            n_heads=num_heads,
            e_layers=enc_layers,
            d_layers=dec_layers,
            d_ff=4 * model_dim,
            dropout=dropout,
            attn=attn,
            embed='fixed',
            freq='h',
            activation=activation,
            output_attention=False,
            distil=distil,
            mix=mix
        )

        # 分类任务额外处理
        if mode == 'classification':
            self.classifier = nn.Sequential(
                nn.Linear(model_dim, model_dim // 2),
                nn.ReLU(),
                nn.Linear(model_dim // 2, num_classes),
                nn.Softmax(dim=-1)
            )
        else:
            self.classifier = None

    def forward(self, src, x_mark_enc=None, x_mark_dec=None):
        """
        src: (batch_size, seq_len, input_dim)
        x_mark_enc: (batch_size, seq_len, time_features) 时间信息（可选）
        x_mark_dec: (batch_size, pred_len + label_len, time_features)（可选）
        
        返回:
        - output: (batch_size, output_dim) 或 (batch_size, output_dim, num_classes)
        """
        batch_size = src.size(0)

        # 构建 decoder 输入（使用 last label_len 个值作为起始）
        if hasattr(self, 'label_len') and self.label_len > 0:
            dec_input = torch.cat([
                src[:, -self.label_len:, :], 
                torch.zeros((batch_size, self.pred_len, src.shape[-1]), device=src.device)
            ], dim=1)
        else:
            dec_input = torch.zeros((batch_size, self.pred_len, src.shape[-1]), device=src.device)

        # Forward through Informer
        output = self.model(
            x_enc=src,
            x_mark_enc=x_mark_enc,
            x_dec=dec_input,
            x_mark_dec=x_mark_dec
        )  # shape: (B, pred_len, output_dim)

        # 如果是分类任务，我们可以对所有预测时间步做平均或取最后一个
        if self.mode == 'classification':
            output = self.classifier(output.mean(dim=1))  # (B, num_classes)
        else:
            # 回归任务返回整个预测序列
            output = output  # (B, pred_len, output_dim)

        return output