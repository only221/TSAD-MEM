import argparse
import json
import os
import torch
import sys
import pandas as pd
import random
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from utils.trainer import Trainer
from models.model import Model
from utils.data_loader import get_data_loader_with_meta  

DEFAULT_LABEL_EVAL_PARAMS: Dict[str, Any] = {
    "lr": 1e-4,
    "individual": True,
    "head_dropout": 0.1,
    "score_lambda": 0.5,
    "auxi_loss": "MAE",
    "auxi_type": "complex",
    "auxi_mode": "fft",
    "auxi_lambda": 0.3,
    "module_first": True,
    "mask": False,
    "inference_patch_size":16,
    "pretrained_model": None,
    "num_epochs": 5,
    "batch_size": 32,
    "inference_patch_stride":1,
    "patience": 3,
    "anomaly_ratio": 3.0,
    "seq_len": 96,
    "pct_start": 0.3,
    "revin": True,
    "affine": True,
    "subtract_last": False,
    "lradj": "type1",
    "time_patch_size": 8,
    "time_patch_stride": 8,
    "time_emb_dim": 256,
    "time_dropout": 0.2,
    "time_e_layers": 3,
    "time_n_heads": 16,
    "time_dim_head": 32,
    "time_d_ff": 256,
    "time_B_num": 10,
    "memory_tau": 0.1,
    "freq_patch_size": 16,
    "freq_patch_stride": 8,
    "freq_emb_dim": 256,
    "freq_e_layers": 3,
    "freq_n_heads": 16,
    "freq_dim_head": 32,
    "freq_d_ff": 256,
    "freq_dropout": 0.2,
    "freq_B_num": 10,
    "freq_decoder_dim": 128,
    "mem_lamda": 0.00,
}

@dataclass
class LabelEvalConfig:
    """Configuration for TSAD-MEM training and evaluation."""

    lr: float = DEFAULT_LABEL_EVAL_PARAMS["lr"]
    individual: bool = DEFAULT_LABEL_EVAL_PARAMS["individual"]
    head_dropout: float = DEFAULT_LABEL_EVAL_PARAMS["head_dropout"]
    score_lambda: float = DEFAULT_LABEL_EVAL_PARAMS["score_lambda"]

    auxi_loss: str = DEFAULT_LABEL_EVAL_PARAMS["auxi_loss"]
    auxi_type: str = DEFAULT_LABEL_EVAL_PARAMS["auxi_type"]
    auxi_mode: str = DEFAULT_LABEL_EVAL_PARAMS["auxi_mode"]
    auxi_lambda: float = DEFAULT_LABEL_EVAL_PARAMS["auxi_lambda"]
    mem_lamda: float = DEFAULT_LABEL_EVAL_PARAMS["mem_lamda"]

    module_first: bool = DEFAULT_LABEL_EVAL_PARAMS["module_first"]
    mask: bool = DEFAULT_LABEL_EVAL_PARAMS["mask"]
    pretrained_model: Optional[str] = DEFAULT_LABEL_EVAL_PARAMS["pretrained_model"]
    num_epochs: int = DEFAULT_LABEL_EVAL_PARAMS["num_epochs"]
    batch_size: int = DEFAULT_LABEL_EVAL_PARAMS["batch_size"]
    inference_patch_size: int = DEFAULT_LABEL_EVAL_PARAMS["inference_patch_size"]
    inference_patch_stride: int = DEFAULT_LABEL_EVAL_PARAMS["inference_patch_stride"]

    patience: int = DEFAULT_LABEL_EVAL_PARAMS["patience"]
    anomaly_ratio: float = DEFAULT_LABEL_EVAL_PARAMS["anomaly_ratio"]
    seq_len: int = DEFAULT_LABEL_EVAL_PARAMS["seq_len"]
    pct_start: float = DEFAULT_LABEL_EVAL_PARAMS["pct_start"]
    revin: bool = DEFAULT_LABEL_EVAL_PARAMS["revin"]
    affine: bool = DEFAULT_LABEL_EVAL_PARAMS["affine"]
    subtract_last: bool = DEFAULT_LABEL_EVAL_PARAMS["subtract_last"]
    lradj: str = DEFAULT_LABEL_EVAL_PARAMS["lradj"]

    time_patch_size: int = DEFAULT_LABEL_EVAL_PARAMS["time_patch_size"]
    time_patch_stride: int = DEFAULT_LABEL_EVAL_PARAMS["time_patch_stride"]
    time_emb_dim: int = DEFAULT_LABEL_EVAL_PARAMS["time_emb_dim"]
    time_dropout: float = DEFAULT_LABEL_EVAL_PARAMS["time_dropout"]
    time_e_layers: int = DEFAULT_LABEL_EVAL_PARAMS["time_e_layers"]
    time_n_heads: int = DEFAULT_LABEL_EVAL_PARAMS["time_n_heads"]
    time_dim_head: int = DEFAULT_LABEL_EVAL_PARAMS["time_dim_head"]
    time_d_ff: int = DEFAULT_LABEL_EVAL_PARAMS["time_d_ff"]
    time_B_num: int = DEFAULT_LABEL_EVAL_PARAMS["time_B_num"]
    memory_tau: float = DEFAULT_LABEL_EVAL_PARAMS["memory_tau"]

    freq_patch_size: int = DEFAULT_LABEL_EVAL_PARAMS["freq_patch_size"]
    freq_patch_stride: int = DEFAULT_LABEL_EVAL_PARAMS["freq_patch_stride"]
    freq_emb_dim: int = DEFAULT_LABEL_EVAL_PARAMS["freq_emb_dim"]
    freq_e_layers: int = DEFAULT_LABEL_EVAL_PARAMS["freq_e_layers"]
    freq_n_heads: int = DEFAULT_LABEL_EVAL_PARAMS["freq_n_heads"]
    freq_dim_head: int = DEFAULT_LABEL_EVAL_PARAMS["freq_dim_head"]
    freq_d_ff: int = DEFAULT_LABEL_EVAL_PARAMS["freq_d_ff"]
    freq_dropout: float = DEFAULT_LABEL_EVAL_PARAMS["freq_dropout"]
    freq_B_num: int = DEFAULT_LABEL_EVAL_PARAMS["freq_B_num"]
    freq_decoder_dim: int = DEFAULT_LABEL_EVAL_PARAMS["freq_decoder_dim"]

    dataset: str = ""
    c_in: int = 0  
    eval_mode: str = field(default="label", init=False)

    def __post_init__(self) -> None:
        pass

def str_to_bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    """Utility to add a boolean flag with both --foo/--no-foo and --foo true/false syntax."""
    parser.add_argument(f"--{name}", type=str_to_bool, nargs='?', const=True, default=default,
                       help=f"Set {name} (default: {default}). Can use --{name}, --{name} true/false")
    parser.add_argument(f"--no-{name}", dest=name, action="store_false",
                       help=f"Disable {name}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TSAD-MEM training and evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

   
    parser.add_argument("--dataset", required=True, help="Dataset name (SMD, SMAP, etc.)")
    parser.add_argument("--data_path", required=True, help="Path to dataset root")
    parser.add_argument("--meta_path", default=str(PROJECT_DIR / "data" / "DETECT_META.csv"),
                        help="CSV containing file_name and train_lens")
    parser.add_argument("--seed", type=int, default=2021, help="Random seed")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="Train and save a checkpoint, or evaluate an existing checkpoint")
    parser.add_argument("--config", default=None, help="Path to JSON config; defaults to configs/<DATASET>.json")
    parser.add_argument("--model_path", help="Checkpoint path (written in train mode, loaded in test mode)")
    parser.add_argument("--save_path", default="results/label_evaluation", help="Directory to save results")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--threshold", type=float, help="Manual threshold for binary classification")
    add_bool_arg(parser, "swanlab", False)
    parser.add_argument("--swanlab_project", default="TSAD-MEM", help="SwanLab project name")
    parser.add_argument("--swanlab_experiment", help="SwanLab experiment name")
    parser.add_argument("--swanlab_mode", choices=["cloud", "local", "offline"], default="cloud",
                        help="SwanLab logging mode")
    parser.add_argument("--swanlab_log_interval", type=int, default=10,
                        help="Log training metrics every N batches")

    for key, value in DEFAULT_LABEL_EVAL_PARAMS.items():
        if isinstance(value, bool):
            add_bool_arg(parser, key, None)
        else:
            parser.add_argument(f"--{key}", type=str if value is None else type(value), default=None)

    return parser.parse_args()


def load_json_config(path: str) -> Dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two config dictionaries (shallow). CLI args always win."""
    merged = base.copy()
    merged.update({k: v for k, v in override.items() if v is not None})
    return merged

def flatten_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested config dictionary."""
    flattened = {}
    for key, value in config_dict.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flattened[nested_key] = nested_value
        else:
            flattened[key] = value
    return flattened

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[INFO] Random seed set to: {seed}")

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config_path = args.config or str(PROJECT_DIR / "configs" / f"{args.dataset}.json")
    file_cfg = load_json_config(config_path)
    flattened_file_cfg = flatten_config(file_cfg)
    flat_cli_cfg: Dict[str, Any] = {k: v for k, v in vars(args).items() if k in DEFAULT_LABEL_EVAL_PARAMS}
    merged_cfg = merge_configs(DEFAULT_LABEL_EVAL_PARAMS, merge_configs(flattened_file_cfg, flat_cli_cfg))
    final_cfg_dict = {k: v for k, v in merged_cfg.items() if k in DEFAULT_LABEL_EVAL_PARAMS}
    cfg = LabelEvalConfig(dataset=args.dataset, **final_cfg_dict)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a compatible PyTorch build or pass --device cpu.")
    device = torch.device(args.device)

    meta_path = Path(args.meta_path)
    train_data, train_label, test_data, test_label = get_data_loader_with_meta(
        data_path=args.data_path,
        meta_path=str(meta_path),
        dataset_name=args.dataset,
        config=cfg,
    )
    

    if isinstance(train_data, pd.DataFrame):
        cfg.c_in = len(train_data.columns)
    else:
        cfg.c_in = train_data.shape[1]




    model = Model(cfg)
    trainer = Trainer(
        model=model,
        train_data=train_data,
        train_label=train_label,
        test_data=test_data,
        test_label=test_label,
        config=cfg,
        device=device,
        save_path=args.save_path,
    )

    model_path = Path(args.model_path or (PROJECT_DIR / "checkpoints" / f"{args.dataset}.pth"))
    if args.mode == "train":
        print("[INFO] Training model…")
        swanlab_client = None
        if args.swanlab:
            os.environ.setdefault(
                "SWANLAB_SAVE_DIR", str(PROJECT_DIR / "swanlog" / ".swanlab")
            )
            try:
                import swanlab as swanlab_client
            except ImportError as exc:
                raise RuntimeError(
                    "SwanLab logging was requested but swanlab is not installed. "
                    "Install it with `pip install -r requirements.txt`."
                ) from exc

            swanlab_client.init(
                project=args.swanlab_project,
                mode=args.swanlab_mode,
                experiment_name=args.swanlab_experiment or (
                    f"{args.dataset}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                ),
                logdir=str(PROJECT_DIR / "swanlog"),
                config={
                    **cfg.__dict__,
                    "dataset": args.dataset,
                    "mode": args.mode,
                    "checkpoint_path": str(model_path),
                },
            )

        try:
            trainer.fit(
                train_data,
                logger=swanlab_client,
                log_interval=args.swanlab_log_interval,
            )
            trainer.save_model(model_path)
            print(f"[INFO] Training complete. Checkpoint saved to {model_path}")
        finally:
            if swanlab_client is not None:
                swanlab_client.finish()
        return

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}. Run with --mode train first, "
            "or pass its path with --model_path."
        )
    trainer.load_model(model_path)
    trainer.prepare_evaluation_data(train_data)

    print("\n[INFO] Evaluating with label-based metrics…")
    
    test_loss, metrics, th = trainer.evaluate_label(
        test_data, test_label, phase="test", threshold=args.threshold
    )

    print(f"\n[RESULTS] Label-based evaluation ({len(metrics)} metrics):")
    for name, value in metrics.items():
        print(f"{name}: {float(value):.6f}")

    print(f"Test Loss: {test_loss:.6f}")
    print(f"Threshold: {th:.6f}")

    results = {
        "test_loss": float(test_loss),
        "eval_mode": cfg.eval_mode,
        "threshold": float(th),
        "metrics": {name: float(value) for name, value in metrics.items()},
        "dataset": args.dataset,
        "seed": args.seed,
        "config": cfg.__dict__,
    }

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"label_results_{args.dataset}.json"
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Results saved → {out_path}")
    


if __name__ == "__main__":
    main()

