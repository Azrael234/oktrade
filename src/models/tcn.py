import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNPredictor(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size, 
                 output_dim=5, dropout=0.1, mode='regression', num_classes=None):
        super().__init__()
        self.mode = mode
        self.output_dim = output_dim
        
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size, dropout)
        
        # 输出层
        if mode == 'regression':
            self.output_layer = nn.Linear(num_channels[-1], 1)  # 每个时间步预测1个值
        else:
            assert num_classes is not None, "分类任务必须指定 num_classes"
            self.output_layer = nn.Sequential(
                nn.Linear(num_channels[-1], num_channels[-1] * 2),
                nn.ReLU(),
                nn.Linear(num_channels[-1] * 2, num_classes),
                nn.Softmax(dim=-1) if num_classes > 1 else nn.Sigmoid()
            )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        features = self.tcn(x)  # (batch, channels, seq_len)
        
        if self.mode == 'regression':
            # 取最后 output_dim 个时间步，每个预测1个值
            output = features[:, :, -self.output_dim:]  # (batch, channels, output_dim)
            output = output.permute(0, 2, 1)  # (batch, output_dim, channels)
            output = self.output_layer(output)  # (batch, output_dim, 1)
            output = output.squeeze(-1)  # (batch, output_dim)
        else:
            # 分类任务：取最后一个时间步
            output = features[:, :, -1]  # (batch, channels)
            output = self.output_layer(output)  # (batch, num_classes)
        
        return output


class TemporalConvNet(nn.Module):
    """时间卷积网络（保持原实现不变）"""
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1)*dilation_size, 
                                   dropout=dropout)]
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class TemporalBlock(nn.Module):
    """时间卷积块（保持原实现不变）"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)