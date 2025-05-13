import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels, kernel_size, dropout=0.1):
        super().__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], output_dim)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
        y = self.tcn(x)
        y = y[:, :, -1]  # 取最后一个时间步
        return self.fc(y)

class TemporalConvNet(nn.Module):
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