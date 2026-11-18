import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .time_path import Time_Path
from .freq_path import Freq_Path
from .utils import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.revin_layer = RevIN(configs.c_in, affine=configs.affine)
        self.fusion_layer = nn.Linear(configs.seq_len * 2, configs.seq_len)
        self.time_path = Time_Path(configs=configs)
        self.freq_path = Freq_Path(configs)  
        
    def forward(self, x):
        # x: [B, seq_len, c_in]
        x = self.revin_layer(x, 'norm')  # RevIN: [B, seq_len, c_in]
        x = x.permute(0, 2, 1)  # [B, c_in, seq_len]

        x_time_hat , mem_loss_time= self.time_path(x)  # [B, c_in, seq_len]
        x_freq_hat_time, x_freq_hat, mem_loss_freq = self.freq_path(x)  # [B, seq_len, c_in], [B, seq_len, c_in]
        
        fused_features = torch.cat([x_time_hat, x_freq_hat_time], dim=-1)  # [B, c_in, 2*seq_len]
        fused_features = self.fusion_layer(fused_features)  # [B, c_in, seq_len]
        output = fused_features.permute(0, 2, 1)
        output = self.revin_layer(output, 'denorm')  # [B, seq_len, c_in]
        mem_loss = mem_loss_time + mem_loss_freq
        
        return output, x_freq_hat, mem_loss