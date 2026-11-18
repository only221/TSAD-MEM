import torch
import torch.nn as nn
from .utils import Transformer, Freq_Memory,Freq_Decoder

class Freq_Path(nn.Module):
    def __init__(self, configs, **kwargs):
        super(Freq_Path, self).__init__()
        self.freq_patch_size = configs.freq_patch_size
        self.freq_patch_stride = configs.freq_patch_stride
        self.seq_len = configs.seq_len
        self.num_patches = int((configs.seq_len - configs.freq_patch_size) // configs.freq_patch_stride + 1)
        self.ircom = nn.Linear(configs.seq_len * 2, configs.seq_len)
        self.dropout = nn.Dropout(configs.freq_dropout)
        self.mlp_head = nn.Linear(configs.freq_emb_dim, configs.freq_decoder_dim)
        self.to_patch_embedding = nn.Sequential(nn.Linear(configs.freq_patch_size * 2, configs.freq_emb_dim), nn.Dropout(configs.freq_dropout))
        self.freq_transformer = Transformer(
            dim=configs.freq_emb_dim, depth=configs.freq_e_layers, heads=configs.freq_n_heads,
            dim_head=configs.freq_dim_head, mlp_dim=configs.freq_d_ff, dropout=configs.freq_dropout
        )
        self.freq_memory = Freq_Memory(B_num=configs.freq_B_num, memo_dim=configs.freq_emb_dim, tau=configs.memory_tau)


        self.freq_decoder = Freq_Decoder(configs)

    def forward(self, x):
        x = torch.fft.fft(x)
        x1, x2 = x.real, x.imag
        x1 = x1.unfold(dimension=-1, size=self.freq_patch_size, step=self.freq_patch_stride)
        x2 = x2.unfold(dimension=-1, size=self.freq_patch_size, step=self.freq_patch_stride)
        x1 = x1.permute(0, 2, 1, 3)
        x2 = x2.permute(0, 2, 1, 3)
        batch_size, patch_num, c_in, patch_size = x1.shape

        x1 = x1.reshape(batch_size * patch_num, c_in, patch_size)
        x2 = x2.reshape(batch_size * patch_num, c_in, patch_size)
        x_cat = torch.cat((x1, x2), -1)

        x_emb = self.to_patch_embedding(x_cat)
        z = self.freq_transformer(x_emb)
        q = z.reshape(batch_size, patch_num, c_in, -1)
        z_reconstructed, mem_loss = self.freq_memory(q)
        if z_reconstructed.shape != z.shape:
            z_reconstructed = z_reconstructed.reshape(z.shape)
        
        z_reconstructed = self.dropout(z_reconstructed)
        z_reconstructed = self.mlp_head(z_reconstructed)
        x_hat_freq = self.freq_decoder(z_reconstructed, batch_size=batch_size, patch_num=patch_num, c_in=c_in)
        x_hat_time = torch.fft.ifft(x_hat_freq)
        zr = x_hat_time.real
        zi = x_hat_time.imag
        x_hat_time = self.ircom(torch.cat((zr, zi), -1))
        
        return x_hat_time, x_hat_freq.permute(0, 2, 1), mem_loss
