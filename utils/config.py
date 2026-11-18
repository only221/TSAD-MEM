class ModelConfig:

    def __init__(self, **kwargs):

        self.seq_len = kwargs.get('seq_len', 100)
        self.c_in = kwargs.get('c_in', 25)
        self.batch_size = kwargs.get('batch_size', 32)
        

        self.lr = kwargs.get('lr', 0.001)
        self.num_epochs = kwargs.get('num_epochs', 100)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.patience = kwargs.get('patience', 10)
        

        self.d_model = kwargs.get('d_model', 512)
        self.n_heads = kwargs.get('n_heads', 8)
        self.e_layers = kwargs.get('e_layers', 3)
        self.dropout = kwargs.get('dropout', 0.1)
        

        self.affine = kwargs.get('affine', True)

        self.subtract_last = kwargs.get('subtract_last', False)
        

        self.time_patch_size = kwargs.get('time_patch_size', 16)
        self.time_patch_stride = kwargs.get('time_patch_stride', 8)  # 添加这个缺失的参数
        self.freq_patch_size = kwargs.get('freq_patch_size', 8)
        self.time_emb_dim = kwargs.get('time_emb_dim', 256)
        self.freq_emb_dim = kwargs.get('freq_emb_dim', 256)
        

        self.memory_size = kwargs.get('memory_size', 100)
        self.memory_tau = kwargs.get('memory_tau', 0.1)
        

        self.recon_weight = kwargs.get('recon_weight', 1.0)
        self.aux_weight = kwargs.get('aux_weight', 0.1)
        

        self.num_workers = kwargs.get('num_workers', 0)
        self.dataset = kwargs.get('dataset', 'Unknown')
        

        self.time_dropout = kwargs.get('time_dropout', 0.2)
        self.time_e_layers = kwargs.get('time_e_layers', 3)
        self.time_n_heads = kwargs.get('time_n_heads', 16)
        self.time_dim_head = kwargs.get('time_dim_head', 32)
        self.time_d_ff = kwargs.get('time_d_ff', 256)
        self.time_B_num = kwargs.get('time_B_num', 10)
        

        self.freq_patch_stride = kwargs.get('freq_patch_stride', 8)
        self.freq_dropout = kwargs.get('freq_dropout', 0.2)
        self.freq_e_layers = kwargs.get('freq_e_layers', 3)
        self.freq_n_heads = kwargs.get('freq_n_heads', 16)
        self.freq_dim_head = kwargs.get('freq_dim_head', 32)
        self.freq_d_ff = kwargs.get('freq_d_ff', 256)
        self.freq_B_num = kwargs.get('freq_B_num', 10)
        self.freq_decoder_dim = kwargs.get('freq_decoder_dim', 128)
    
    def __repr__(self):
        return f"ModelConfig({self.__dict__})"


def validate_metrics_config(metrics_list, eval_mode):

    if eval_mode == 'label':
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f_score',
            'adjust_accuracy', 'adjust_precision', 'adjust_recall', 'adjust_f_score',
            'rrecall', 'rprecision', 'precision_at_k', 'rf', 
            'affiliation_f', 'affiliation_precision', 'affiliation_recall'
        ]
        missing_metrics = set(expected_metrics) - set(metrics_list)
        if missing_metrics:
            print(f"Warning: Missing metrics in config: {missing_metrics}")
    return True