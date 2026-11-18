import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from torch import einsum
from einops import rearrange

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):

        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        elif mode == 'transform':
            x = self._normalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.d_k = math.sqrt(dim_head)
        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        scale = 1 / self.d_k

        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        # Compute attention scores
        scores = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        # Apply attention
        attn = self.attend(scores)
        attn = self.dropout(attn)  
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) 

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.8):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x_attn = attn(x) 
            x = x_attn + x
            x = ff(x) + x
        return x

def entropy_loss(attn_probs):
    ent = -(attn_probs * (attn_probs + 1e-8).log()).sum(-1)   # [... ]
    return ent.mean()

class Time_Memory(nn.Module):
    def __init__(self, *, B_num: int, memo_dim: int, emb_dim: int,
                 tau: float = 0.1, dropout: float = 0.1,
                 beta_rec: float = 1.0, beta_entropy: float = 1.0,
                 beta_diversity: float = 1.0):
        super().__init__()
        self.B_num = B_num
        self.memo_dim = memo_dim
        self.emb_dim = emb_dim
        self.tau = tau
        self.dropout = nn.Dropout(dropout)
        
        self.beta_rec = beta_rec
        self.beta_entropy = beta_entropy
        self.beta_diversity = beta_diversity
        
        self.patch_prototypes = nn.Parameter(
            torch.randn(1, B_num, memo_dim) / memo_dim**0.5  
        )
        
        self.U1 = nn.Linear(memo_dim, memo_dim, bias=False)  # 论文中 U1^z
        self.U2 = nn.Linear(memo_dim, memo_dim, bias=False)  # 论文中 U2^z
    
    def forward(self, q):

        B, P, _ = q.shape
        

        q_feat = q
        patch_prototypes = self.patch_prototypes.expand(B, -1, -1)

        sim_update = torch.bmm(patch_prototypes, q_feat.transpose(1, 2)) / self.tau
        attn_update = self.dropout(F.softmax(sim_update, dim=-1))

        weighted_q = torch.bmm(attn_update, q_feat)
        

        gate = torch.sigmoid(
            self.U1(patch_prototypes) + self.U2(weighted_q)
        )
        

        updated_prototypes = (1 - gate) * patch_prototypes + gate * weighted_q  # [B, B_num, memo_dim]，更新后的 B^z
        

        sim_recon = torch.bmm(q_feat, updated_prototypes.transpose(1, 2)) / self.tau  # [B, P, B_num]，重构相似度
        attn_recon = self.dropout(F.softmax(sim_recon, dim=-1))
        z_recon = torch.bmm(attn_recon, updated_prototypes)
        
        loss_rec = F.mse_loss(z_recon, q_feat)
        

        loss_ent = -torch.sum(attn_recon * torch.log(attn_recon + 1e-8), dim=-1).mean()

        proto_norm = F.normalize(updated_prototypes, dim=-1)
        sim_matrix = torch.bmm(proto_norm, proto_norm.transpose(1, 2))
        mask = torch.triu(torch.ones(self.B_num, self.B_num, device=sim_matrix.device), diagonal=1)
        diversity_loss = (sim_matrix * mask.unsqueeze(0)).sum() / (self.B_num * (self.B_num - 1) / 2)
        
        mem_loss = self.beta_rec * loss_rec + self.beta_entropy * loss_ent + self.beta_diversity * diversity_loss
        
        return z_recon, mem_loss


class Time_Decoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        time_emb_dim: int,
        seq_len: int,
        patch_size: int,
        patch_stride: int,
        dropout: float = 0.1,
    ):

        super().__init__()
        assert patch_stride == patch_size

        self.c_in       = c_in
        self.seq_len    = seq_len
        self.patch_size = patch_size
        self.num_patches = math.ceil(seq_len / patch_size)

        self.linear  = nn.Linear(time_emb_dim, c_in * patch_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x : [B, P, time_emb_dim]
        """
        B, P, _ = x.shape
        assert P == self.num_patches

        x = self.dropout(self.linear(x))               # [B, P, C*K]
        x = x.view(B, P, self.c_in, self.patch_size)   # [B, P, C, K]
        x = x.permute(0, 2, 1, 3).contiguous()         # [B, C, P, K]
        x = x.view(B, self.c_in, -1)                   # [B, C, P*K]
        return x[..., :self.seq_len]
class Freq_Memory(nn.Module):
    def __init__(self, *, B_num: int, memo_dim: int,
                 tau: float = 0.1, dropout: float = 0.1,
                 beta_rec: float = 1.0, beta_entropy: float = 1.0,
                 lambda_time: float = 0.01, gamma_contrib: float = 0.1,
                 beta_diversity: float = 1.0):
        super().__init__()
        self.B_num, self.memo_dim, self.tau = B_num, memo_dim, tau
        self.beta_rec, self.beta_entropy ,self.beta_diversity= beta_rec, beta_entropy,beta_diversity

        self.patch_prototypes = nn.Parameter(
            torch.randn(1, B_num, memo_dim) / memo_dim**0.5  
        ) 
        self.U1 = nn.Linear(memo_dim, memo_dim, bias=False)  
        self.U2 = nn.Linear(memo_dim, memo_dim, bias=False)
 
        self.dropout = nn.Dropout(dropout)

    def forward(self, q):

        B, P, C, D = q.shape

        patch_prototypes = self.patch_prototypes.expand(B, -1, -1)  
        z_patches = []
        loss_rec, loss_ent, loss_div = 0., 0., 0.

        for p in range(P):
            q_patch = q[:, p]                    # [B,C,D]
            sim_update = torch.bmm(patch_prototypes, q_patch.transpose(1, 2)) / self.tau  # [B, B_num, P]，相似度矩阵 V^z
            attn_update = self.dropout(F.softmax(sim_update, dim=-1))  # [B, B_num, P]，注意力权重 v_ij^z
            
            weighted_q = torch.bmm(attn_update, q_patch)  # [B, B_num, memo_dim]

            gate = torch.sigmoid(self.U1(patch_prototypes) + self.U2(weighted_q) )  # [B, B_num, memo_dim]
            updated_prototypes = (1 - gate) * patch_prototypes + gate * weighted_q  # [B, B_num, memo_dim]

            sim_recon = torch.bmm(q_patch, updated_prototypes.transpose(1, 2)) / self.tau  # [B, C, B_num]
            attn_recon = self.dropout(F.softmax(sim_recon, dim=-1))  # [B, C, B_num]
            z_recon = torch.bmm(attn_recon, updated_prototypes)  # [B, C, memo_dim]
            

            z_patches.append(z_recon)
            
            loss_rec += F.mse_loss(z_recon, q_patch)
            loss_ent += entropy_loss(attn_recon)

            proto_norm = F.normalize(updated_prototypes, dim=-1)
            sim_matrix = torch.bmm(proto_norm, proto_norm.transpose(1, 2))  # [B, B_num, B_num]
            mask = torch.triu(torch.ones(self.B_num, self.B_num, device=sim_matrix.device), diagonal=1)
            loss_div += (sim_matrix * mask.unsqueeze(0)).sum() / ( self.B_num * (self.B_num - 1) / 2)

        loss_rec = loss_rec / P
        loss_ent = loss_ent / P
        loss_div = loss_div / P
        mem_loss = self.beta_rec*loss_rec + self.beta_entropy*loss_ent + self.beta_diversity*loss_div

        z_recon = torch.stack(z_patches, dim=1)            # [B,P,C,D]
        z_recon_flat = rearrange(z_recon, 'b p c d -> (b p) c d')

        return z_recon_flat, mem_loss
    

class Freq_Decoder(nn.Module):
    """
    parameters:
    seq_len: int, length of sequence
    freq_patch_size: int, size of frequency patch
    freq_patch_stride: int, stride of frequency patch
    freq_decoder_dim: int, dimension of frequency decoder
    individual: bool, whether to use individual head
    c_in: int, number of channels
    head_dropout: float, dropout rate
    """
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.num_patches = int((configs.seq_len - configs.freq_patch_size) / configs.freq_patch_stride + 1)
        self.head_nf_f = configs.freq_decoder_dim * self.num_patches
        self.get_r = nn.Linear(configs.freq_decoder_dim, configs.freq_decoder_dim)  
        self.get_i = nn.Linear(configs.freq_decoder_dim, configs.freq_decoder_dim)
        self.head_f1 = Flatten_Head(configs.individual, configs.c_in, self.head_nf_f, configs.seq_len, head_dropout=configs.head_dropout)
        self.head_f2 = Flatten_Head(configs.individual, configs.c_in, self.head_nf_f, configs.seq_len, head_dropout=configs.head_dropout)


    def forward(self, z_reconstructed, batch_size, patch_num, c_in):
        z1 = self.get_r(z_reconstructed)
        z2 = self.get_i(z_reconstructed)
        z1 = torch.reshape(z1, (batch_size, patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size, patch_num, c_in, z2.shape[-1]))
        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        x_hat_freq1 = self.head_f1(z1)
        x_hat_freq2 = self.head_f2(z2)
        x_hat_freq = torch.complex(x_hat_freq1, x_hat_freq2)
        return x_hat_freq

class Flatten_Head(nn.Module):
    def __init__(self, individual, c_in, nf, seq_len, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.c_in = c_in
        self.nf = nf
        self.seq_len = seq_len


        if self.individual:
            self.linears1 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.c_in):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, seq_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, seq_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.c_in):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears1[i](z)  # z: [bs x seq_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x seq_len]
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)

        return x