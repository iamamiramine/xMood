from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import os
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from application.classifier.models.model.net import SAN
from application.classifier.models.task.pipeline import PEmoPipeline
from application.classifier.models.task.runner import Runner
import wandb

from application.encoder.models.vocab_model import RemiVocab


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_config(args: Namespace) -> DictConfig:
    parent_config_dir = Path("shared/conf/")
    child_config_dir = parent_config_dir / args.dataset
    task_config_dir = child_config_dir / "task"
    midi_config_dir = child_config_dir / "midi"

    config = OmegaConf.create()
    task_config = OmegaConf.load(task_config_dir / f"{args.task}.yaml")
    midi_config = OmegaConf.load(midi_config_dir / f"{args.midi}.yaml")

    # Add labels_path to task config if specified
    if args.labels_path:
        task_config.labels_path = args.labels_path

    config.update(task=task_config, midi=midi_config, hparams=vars(args))
    return config


def get_tensorboard_logger(args: Namespace) -> TensorBoardLogger:
    logger = TensorBoardLogger(save_dir=f"exp/{args.dataset}", name=args.task, version=f"{args.midi}/")
    return logger


def get_wandb_logger(model):
    logger = WandbLogger()
    logger.watch(model)
    return logger


def get_checkpoint_callback(args, save_path) -> ModelCheckpoint:
    prefix = save_path
    suffix = "Best-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=prefix,
        filename=suffix,
        save_top_k=1,
        save_last=True,
        monitor="val_acc",
        mode="max",
        save_weights_only=True,
        verbose=True,
    )
    return checkpoint_callback


def get_best_performance(runner, save_path):
    for fnames in os.listdir(save_path):
        if "Best" in fnames:
            checkpoint_path = Path(save_path, fnames)
    state_dict = torch.load(checkpoint_path)
    runner.load_state_dict(state_dict.get("state_dict"))
    return runner


def get_early_stop_callback(args: Namespace) -> EarlyStopping:
    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=5, verbose=True, mode="max")
    return early_stop_callback


def main(
    dataset: str = "EMOPIA",
    midi: str = "magenta",
    task: str = "ar_va",
    labels_path: str = None,
    r: int = 14,
    lstm_hidden_dim: int = 64,  # 128,
    embedding_size: int = 100,  # 300,
    batch_size: int = 4,
    num_workers: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    T_0: int = 45,
    max_epochs: int = 200,
    gpus: int = 1,
    distributed_backend: str = "dp",
    deterministic: bool = True,
    benchmark: bool = False,
    reproduce: bool = True,
) -> None:
    # Convert parameters to an args-like object for compatibility with existing functions
    class Args:
        pass

    args = Args()
    args.dataset = dataset
    args.midi = midi
    args.task = task
    args.labels_path = labels_path
    args.r = r
    args.lstm_hidden_dim = lstm_hidden_dim
    args.embedding_size = embedding_size
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.lr = lr
    args.weight_decay = weight_decay
    args.T_0 = T_0
    args.max_epochs = max_epochs
    args.gpus = gpus
    args.distributed_backend = distributed_backend
    args.deterministic = deterministic
    args.benchmark = benchmark
    args.reproduce = reproduce

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if reproduce:
        seed_everything(42)

    save_path = f"output/demos/demo_2/classifier/{dataset}/{task}/{midi}/batch:{batch_size}-h:{lstm_hidden_dim}-emd:{embedding_size}-wd:{weight_decay}-attn:{r}-T_0:{T_0}-lr:{lr}/"
    fix_config = get_config(args)
    pipeline = PEmoPipeline(config=args, fix_config=fix_config)

    OmegaConf.save(config=fix_config, f=Path(save_path, "hparams.yaml"))

    vocab = RemiVocab()

    # Set cls_type to MOOD if using a labels_path
    cls_type = "MOOD" if labels_path else task

    model = SAN(r=r, num_of_dim=fix_config.task.num_of_dim, vocab_size=len(vocab) + 1, embedding_size=embedding_size, cls_type=cls_type)
    runner = Runner(model, args, eval_type="last")

    # logger = get_wandb_logger(model)
    checkpoint_callback = get_checkpoint_callback(args, save_path)
    early_stop_callback = get_early_stop_callback(args)

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else None,
        benchmark=benchmark,
        deterministic=deterministic,
        # logger=logger,
        callbacks=[
            # early_stop_callback,
            checkpoint_callback
        ],
    )

    trainer.fit(runner, datamodule=pipeline)
    trainer.test(runner, datamodule=pipeline)
    results = {"last": runner.test_results}

    best_runner = Runner(model, args, eval_type="best")
    best_runner = get_best_performance(best_runner, save_path)
    trainer.test(best_runner, datamodule=pipeline)
    results.update({"best": best_runner.test_results})

    with open(Path(save_path, "results_last.json"), mode="w") as io:
        json.dump(results, io, indent=4)

    OmegaConf.save(config=fix_config, f=Path(save_path, "hparams.yaml"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="EMOPIA", type=str)
    parser.add_argument("--midi", default="magenta", type=str)
    parser.add_argument("--task", default="ar_va", type=str)
    parser.add_argument("--labels_path", default=None, type=str, help="Path to CSV file with mood labels")
    # model
    parser.add_argument("--r", default=14, type=int)
    parser.add_argument("--lstm_hidden_dim", default=128, type=float)
    parser.add_argument("--embedding_size", default=300, type=float)
    # pipeline
    parser.add_argument("--batch_size", default=8, type=float)
    parser.add_argument("--num_workers", default=8, type=float)
    # runner
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--T_0", default=45, type=int)
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--distributed_backend", default="dp", type=str)
    parser.add_argument("--deterministic", default=True, type=str2bool)
    parser.add_argument("--benchmark", default=False, type=str2bool)
    parser.add_argument("--reproduce", default=True, type=str2bool)

    args = parser.parse_args()
    main(args)
