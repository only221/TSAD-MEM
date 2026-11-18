import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from typing import Union, Tuple
import time
from .data_loader import anomaly_detection_data_provider
from .early_stopping import EarlyStopping,adjust_learning_rate
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler
from .metrics import Metrics
from .losses import frequency_loss,frequency_criterion
class Trainer:
    def __init__(self, model, train_data, train_label, test_data, test_label, config, device, save_path):
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.scaler = StandardScaler()
        self.criterion = nn.MSELoss()
        self.auxi_loss = frequency_loss(self.config)

        self.seq_len = self.config.seq_len
        self.metrics_calculator = Metrics()
        self.eval_mode = getattr(config, 'eval_mode', 'score')
        self.optimizer = None

    
    def detect_validate(self, valid_data_loader, criterion1, criterion2):
        config = self.config
        total_loss1 = []
        total_loss2 = []
        total_loss3 = []
        self.model.eval()
        device = self.device

        with torch.no_grad():
            for input, _ in valid_data_loader:
                input = input.to(device)
                norm_input = self.model.revin_layer(input, 'norm')
                output, fre , mem_loss= self.model(input)
                output = output.detach().cpu()
                fre = fre.detach().cpu()
                true = input.detach().cpu()
                norm_input =norm_input.detach().cpu()

                loss1 = criterion1(output, true).detach().cpu().numpy()
                total_loss1.append(loss1)
                loss2 = criterion2(fre, norm_input).detach().cpu().numpy()
                total_loss2.append(loss2)
                mem_loss = mem_loss.detach().cpu().numpy()
                total_loss3.append(mem_loss)

        total_loss1 = np.mean(total_loss1)
        total_loss2 = np.mean(total_loss2)
        total_loss3 = np.mean(total_loss3)
        total_loss = total_loss1 + total_loss2 + total_loss3
        
        self.model.train()
        return total_loss    

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        try:
            freq = pd.infer_freq(train_data.index)
        except Exception as ignore:
            freq = 'S'
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]    
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = 48

    def fit(self, train_data, logger=None, log_interval=10):
        if logger is not None and log_interval < 1:
            raise ValueError("log_interval must be at least 1")
        self.detect_hyper_param_tune(train_data)           
        setattr(self.config, "task_name", "anomaly_detection")  
        self.config.c_in = train_data.shape[1]             
        self.model.to(self.device)                         

        config = self.config
        train_data_value, _ = Trainer.train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)
        self._prepare_loaders(train_data)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {total_params}")

        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        main_params = [param for name, param in self.model.named_parameters() if 'mask_generator' not in name]
        self.optimizer = torch.optim.Adam(main_params,
                                          lr=self.config.lr)

        train_steps = len(self.train_data_loader)
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.lr,
        )
        
        time_now = time.time()
        global_step = 0
        for epoch in range(self.config.num_epochs):
            iter_count = 0
            train_loss = []
            rec_losses = []
            auxi_losses = []
            mem_losses = []

            epoch_time = time.time()
            self.model.train()

            for i, (input, target) in enumerate(self.train_data_loader):
                iter_count += 1
                self.optimizer.zero_grad()

                input = input.float().to(self.device)

                output, output_complex, mem_loss = self.model(input)


                rec_loss = self.criterion(output, input)

                norm_input = self.model.revin_layer(input, 'transform')
                auxi_loss = self.auxi_loss(output_complex, norm_input)

                loss = rec_loss +  config.auxi_lambda * auxi_loss + config.mem_lamda * mem_loss

                train_loss.append(loss.item())
                rec_losses.append(rec_loss.item())
                auxi_losses.append(auxi_loss.item())
                mem_losses.append(mem_loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | training time loss: {2:.7f} | training fre loss: {3:.7f} | training mem loss: {4:.7f} ".format(
                            i + 1, epoch + 1, rec_loss.item(), auxi_loss.item(), mem_loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                            (self.config.num_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()
                global_step += 1
                if logger is not None and global_step % log_interval == 0:
                    logger.log({
                        "train/step_loss": loss.item(),
                        "train/step_reconstruction_loss": rec_loss.item(),
                        "train/step_frequency_loss": auxi_loss.item(),
                        "train/step_memory_loss": mem_loss.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    }, step=global_step)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            valid_loss = self.detect_validate(self.valid_data_loader, self.criterion, self.auxi_loss)  #验证损失
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, valid_loss
                )
            )
            if logger is not None:
                logger.log({
                    "epoch": epoch + 1,
                    "train/epoch_loss": float(train_loss),
                    "train/reconstruction_loss": float(np.mean(rec_losses)),
                    "train/frequency_loss": float(np.mean(auxi_losses)),
                    "train/memory_loss": float(np.mean(mem_losses)),
                    "validation/loss": float(valid_loss),
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                }, step=global_step)


            self.early_stopping(valid_loss, self.model)   
            #'Validation loss decreased (inf --> 0.475462).  Saving model ...'
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, scheduler, epoch + 1, self.config)  #Updating learning rate to 0.0001

        self.model.load_state_dict(self.early_stopping.check_point)

    def _prepare_loaders(self, train_data):
        train_data_value, valid_data = Trainer.train_val_split(train_data, 0.8, None)
        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )
        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )
        self.train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="train",
        )
        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="val",
        )

    def prepare_evaluation_data(self, train_data):
        if not hasattr(self.scaler, "mean_"):
            raise RuntimeError("Load a trained model before preparing evaluation data")
        self._prepare_loaders(train_data)
    

    def evaluate_label(self, test_data, test_label, phase="test", threshold=None, return_scores=False):
        if not hasattr(self, "train_data_loader"):
            raise RuntimeError("Prepare training data before evaluation")
        test_data = pd.DataFrame(
            self.scaler.transform(test_data.values), columns=test_data.columns, index=test_data.index
        )
 
        self.test_data_loader = anomaly_detection_data_provider(
            test_data,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="test",
        )

        self.thre_loader = anomaly_detection_data_provider(
            test_data,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="thre",
        )
        attens_energy = []

        self.model.to(self.device)
        self.model.eval()
        self.temp_anomaly_criterion = nn.MSELoss(reduction='none')
        self.freq_anomaly_criterion = frequency_criterion(self.config)
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.train_data_loader):     
                batch_x = batch_x.float().to(self.device)
                outputs, _ , mem_loss= self.model(batch_x)
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score + self.config.score_lambda * freq_score).detach().cpu().numpy() 
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        
        attens_energy = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.test_data_loader):     
                batch_x = batch_x.float().to(self.device)
                outputs, _, mem_loss = self.model(batch_x)
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score + self.config.score_lambda * freq_score).detach().cpu().numpy() 
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        attens_energy = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.thre_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.to(self.device)
                outputs, _ ,_ = self.model(batch_x)
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score + self.config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)
                 
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        thr_energy = np.array(attens_energy)
        actual_label = test_label.to_numpy().flatten()

        avg_loss = np.mean(test_energy)
        if return_scores:
            return avg_loss, combined_energy, thr_energy, actual_label

        ratio = self.config.anomaly_ratio
        if isinstance(ratio, list):
            ratio = ratio[0]
        threshold = float(threshold) if threshold is not None else np.percentile(combined_energy, 100 - ratio)
        y_pred_labels = (thr_energy > threshold).astype(int)

        remaining_length = len(actual_label) - len(y_pred_labels)
        if remaining_length > 0:
            y_pred_labels = np.pad(y_pred_labels, (0, remaining_length), mode="constant")

        metrics_results = self.metrics_calculator.evaluate_all_metrics_label(actual_label, y_pred_labels)
        

        return avg_loss, metrics_results, threshold

    def save_model(self, path):
        if not hasattr(self.scaler, "mean_"):
            raise RuntimeError("Cannot save a model before fitting the training scaler")
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': torch.as_tensor(self.scaler.mean_),
            'scaler_scale': torch.as_tensor(self.scaler.scale_),
            'scaler_var': torch.as_tensor(self.scaler.var_),
        }
        path = str(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'scaler_mean' in checkpoint:
            self.scaler.mean_ = checkpoint['scaler_mean'].cpu().numpy()
            self.scaler.scale_ = checkpoint['scaler_scale'].cpu().numpy()
            self.scaler.var_ = checkpoint['scaler_var'].cpu().numpy()
            self.scaler.n_features_in_ = len(self.scaler.mean_)
        else:
            raise ValueError(
                "Checkpoint does not contain scaler parameters; retrain with this version "
                "or provide a checkpoint that includes scaler_mean/scaler_scale/scaler_var."
            )
        print(f"Model loaded from {path}")
    

    @staticmethod
    def train_val_split(train_data, ratio, seq_len):
        if ratio == 1:
            return train_data, None

        elif seq_len is not None:
            border = int((train_data.shape[0]) * ratio)

            train_data_value, valid_data_rest = Trainer.split_before(train_data, border)
            train_data_rest, valid_data = Trainer.split_before(train_data, border - seq_len)
            return train_data_value, valid_data
        else:
            border = int((train_data.shape[0]) * ratio)

            train_data_value, valid_data_rest = Trainer.split_before(train_data, border)
            return train_data_value, valid_data_rest
    
    @staticmethod
    def split_before(
        data: Union[pd.DataFrame, np.ndarray], index: int
     ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[np.ndarray, np.ndarray]]:
        """
        Split time series data into two parts at the specified index.

        :param data: Time series data to be segmented.
        Can be a pandas DataFrame or a NumPy array.
        :param index: Split index position.
        :return: Tuple containing the first and second parts of the data.
        """
        if isinstance(data, pd.DataFrame):
            return data.iloc[:index, :], data.iloc[index:, :]
        elif isinstance(data, np.ndarray):
            return data[:index, :], data[index:, :]
        else:
            raise TypeError("Input data must be a pandas DataFrame or a NumPy array.")
