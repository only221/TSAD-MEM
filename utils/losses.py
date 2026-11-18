import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

class ReconstructionLoss(nn.Module):

    def __init__(self, loss_type='mse'):
        super(ReconstructionLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_type == 'huber':
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, pred, target):
        return self.criterion(pred, target)

class frequency_loss(torch.nn.Module):
    def __init__(self, configs, keep_dim=False, dim=None):
        super(frequency_loss, self).__init__()
        self.keep_dim = keep_dim
        self.dim = dim
        if configs.auxi_mode == "fft":
            self.fft = torch.fft.fft
        elif configs.auxi_mode == "rfft":
            self.fft = torch.fft.rfft
        else:
            raise NotImplementedError
        self.configs = configs
        self.mask = None

    def forward(self, outputs, batch_y):
        if outputs.is_complex():
            frequency_outputs = outputs
        else:
            frequency_outputs = self.fft(outputs, dim=1)
        if self.configs.auxi_type == 'complex':
            loss_auxi = frequency_outputs - self.fft(batch_y, dim=1)
        elif self.configs.auxi_type == 'complex-phase':
            loss_auxi = (frequency_outputs - self.fft(batch_y, dim=1)).angle()
        elif self.configs.auxi_type == 'complex-mag-phase':
            loss_auxi_mag = (frequency_outputs - self.fft(batch_y, dim=1)).abs()
            loss_auxi_phase = (frequency_outputs - self.fft(batch_y, dim=1)).angle()
            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
        elif self.configs.auxi_type == 'phase':
            loss_auxi = frequency_outputs.angle() - self.fft(batch_y, dim=1).angle()
        elif self.configs.auxi_type == 'mag':
            loss_auxi = frequency_outputs.abs() - self.fft(batch_y, dim=1).abs()
        elif self.configs.auxi_type == 'mag-phase':
            loss_auxi_mag = frequency_outputs.abs() - self.fft(batch_y, dim=1).abs()
            loss_auxi_phase = frequency_outputs.angle() - self.fft(batch_y, dim=1).angle()
            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
        else:
            raise NotImplementedError

        if self.configs.auxi_loss == "MAE":
            loss_auxi = loss_auxi.abs().mean(dim=self.dim,
                                             keepdim=self.keep_dim) if self.configs.module_first else loss_auxi.mean(
                dim=self.dim, keepdim=self.keep_dim).abs()  # check the dim of fft
        elif self.configs.auxi_loss == "MSE":
            loss_auxi = (loss_auxi.abs() ** 2).mean(dim=self.dim,
                                                    keepdim=self.keep_dim) if self.configs.module_first else (
                    loss_auxi ** 2).mean(dim=self.dim, keepdim=self.keep_dim).abs()
        else:
            raise NotImplementedError
        return loss_auxi

class frequency_criterion(torch.nn.Module):
    def __init__(self, configs):
        super(frequency_criterion, self).__init__()
        self.metric = frequency_loss(configs, dim=1, keep_dim=True)
        self.patch_size = configs.inference_patch_size
        self.patch_stride = configs.inference_patch_stride
        self.win_size = configs.seq_len
        self.patch_num = int((self.win_size - self.patch_size) / self.patch_stride + 1)
        self.padding_length = self.win_size - (self.patch_size + (self.patch_num - 1) * self.patch_stride)

    def forward(self, outputs, batch_y):

        output_patch = outputs.unfold(dimension=1, size=self.patch_size,
                                      step=self.patch_stride)

        b, n, c, p = output_patch.shape
        output_patch = rearrange(output_patch, 'b n c p -> (b n) p c')
        y_patch = batch_y.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        y_patch = rearrange(y_patch, 'b n c p -> (b n) p c')

        main_part_loss = self.metric(output_patch, y_patch)
        main_part_loss = main_part_loss.repeat(1, self.patch_size, 1)
        main_part_loss = rearrange(main_part_loss, '(b n) p c -> b n p c', b=b)

        end_point = self.patch_size + (self.patch_num - 1) * self.patch_stride - 1
        start_indices = np.array(range(0, end_point, self.patch_stride))
        end_indices = start_indices + self.patch_size

        indices = torch.tensor([range(start_indices[i], end_indices[i]) for i in range(n)]).unsqueeze(0).unsqueeze(-1)
        indices = indices.repeat(b, 1, 1, c).to(main_part_loss.device)
        main_loss = torch.zeros((b, n, self.win_size - self.padding_length, c)).to(main_part_loss.device)
        main_loss.scatter_(dim=2, index=indices, src=main_part_loss)

        # Count covering patches, including patches with exactly zero error.
        coverage = main_loss.new_zeros(self.win_size - self.padding_length)
        patch_indices = indices[0, :, :, 0].reshape(-1)
        coverage.scatter_add_(
            0, patch_indices, main_loss.new_ones(patch_indices.numel())
        )
        main_loss = main_loss.sum(1) / coverage.view(1, -1, 1)

        if self.padding_length > 0:
            padding_loss = self.metric(outputs[:, -self.padding_length:, :], batch_y[:, -self.padding_length:, :])
            padding_loss = padding_loss.repeat(1, self.padding_length, 1)
            total_loss = torch.concat([main_loss, padding_loss], dim=1)
        else:
            total_loss = main_loss
        return total_loss

class CombinedLoss(nn.Module):
    def __init__(self, recon_weight=1.0, auxi_lambda=0.1, loss_type='mse'):
        super(CombinedLoss, self).__init__()
        self.recon_weight = recon_weight
        self.auxi_lambda = auxi_lambda
        
        self.recon_loss = ReconstructionLoss(loss_type)
        self.auxi_loss = frequency_loss(self.config)
    
    def forward(self, time_recon, freq_recon, target):

        main_recon_loss = self.recon_loss(time_recon, target)
        norm_input = self.model.revin_layer(target, 'transform')
        auxi_loss_val = self.aux_loss( freq_recon, norm_input)
        total_loss = self.recon_weight * main_recon_loss + self.auxi_lambda * auxi_loss_val
        
        loss_details = {
            'total_loss': total_loss.item(),
            'recon_loss': main_recon_loss.item(),
            'auxi_loss': auxi_loss_val.item(),
        }
        
        return total_loss, loss_details

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced anomaly detection"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss