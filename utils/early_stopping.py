import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import copy

plt.switch_backend('agg')

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.check_point = copy.deepcopy(model.state_dict())
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):

    learning_rate = getattr(args, 'learning_rate', None) or getattr(args, 'lr', 0.001)
    
    if args.lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: learning_rate if epoch < 3 else learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: learning_rate if epoch < 20 else learning_rate * (0.5 ** ((epoch // 20) // 1))}
    elif args.lradj == 'type5':
        lr_adjust = {epoch: learning_rate if epoch < 10 else learning_rate * (0.5 ** ((epoch // 10) // 1))}
    elif args.lradj == 'type6':
        lr_adjust = {20: learning_rate * 0.5 , 40: learning_rate * 0.01, 60: learning_rate * 0.01, 8: learning_rate * 0.01, 100: learning_rate * 0.01}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: learning_rate if epoch < 10 else learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: learning_rate if epoch < 15 else learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: learning_rate if epoch < 25 else learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: learning_rate if epoch < 5 else learning_rate*0.1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)