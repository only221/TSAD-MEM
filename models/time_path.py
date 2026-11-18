import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import Transformer, Time_Memory, Time_Decoder

class Time_Path(nn.Module):
    def __init__(self, configs):
        super(Time_Path, self).__init__()
        self.patch_size = configs.time_patch_size
        self.patch_stride = configs.time_patch_stride
        self.seq_len = configs.seq_len
        self.num_patches = (configs.seq_len - configs.time_patch_size) // configs.time_patch_stride + 1
        
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(configs.c_in, configs.time_emb_dim),
            nn.Dropout(configs.time_dropout)
        )

        self.pos_embedding = self.create_positional_encoding(self.num_patches, configs.time_emb_dim)
        self.time_transformer = Transformer(
            dim=configs.time_emb_dim,
            depth=configs.time_e_layers,
            heads=configs.time_n_heads,
            dim_head=configs.time_dim_head,
            mlp_dim=configs.time_d_ff,
            dropout=configs.time_dropout
        )
        self.time_memory = Time_Memory(
            B_num=configs.time_B_num,
            memo_dim=configs.time_emb_dim,
            emb_dim=configs.time_emb_dim,
            tau=configs.memory_tau,
            dropout=configs.time_dropout
        )
        self.decoder = Time_Decoder(
            c_in=configs.c_in,
            time_emb_dim=configs.time_emb_dim,
            seq_len=configs.seq_len,
            patch_size=configs.time_patch_size,
            patch_stride=configs.time_patch_stride,
            dropout=configs.time_dropout
        )

    def create_positional_encoding(self, num_patches, emb_dim):
        position = torch.arange(num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2, dtype=torch.float) * (-math.log(10000.0) / emb_dim))
        pe = torch.zeros(1, num_patches, emb_dim)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe, requires_grad=False)

    def patch_average(self, x, size):
        seq_len = x.shape[2]
        if seq_len % self.patch_stride != 0:
            pad_len = self.patch_stride - (seq_len % self.patch_stride)
            x = F.pad(x, (0, pad_len))
        x_patched = x.unfold(dimension=2, size=size, step=self.patch_stride)
        x_patched = x_patched.mean(dim=-1)
        x_patched = x_patched.permute(0, 2, 1)
        return x_patched

    def forward(self, x):
        x_z = self.patch_average(x, self.patch_size)
        x_emb = self.to_patch_embedding(x_z)
        x_emb = x_emb + self.pos_embedding
        z = self.time_transformer(x_emb)
        z_reconstructed, mem_loss = self.time_memory(z)
        x_hat_time = self.decoder(z_reconstructed)
        return x_hat_time, mem_loss