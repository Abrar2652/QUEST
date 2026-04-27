"""
QUEST training script.

Runs from the QUEST/ directory but uses unKR_ref/ for source code and datasets.
"""

import sys
import os
import argparse

QUEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(QUEST_DIR, "src"))

import random
import numpy as np
import torch
torch.set_num_threads(30)
import pytorch_lightning as pl
import torch.autograd
from pytorch_lightning import seed_everything
from unKR.utils import *
from unKR.data.Sampler import *


torch.autograd.set_detect_anomaly(True)


def main():
    # Parse --gpu separately before PL's parser
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--gpu", type=int, default=0)
    pre_args, remaining = pre_parser.parse_known_args()
    gpu_id = pre_args.gpu

    with torch.autograd.set_detect_anomaly(True):
        parser = setup_parser()
        args = parser.parse_args(remaining)
        if args.load_config:
            args = load_config(args, args.config_path)
        args.gpu = f"cuda:{gpu_id}"
        seed_everything(args.seed)

        # Resolve dataset path relative to QUEST dir
        if not os.path.isabs(args.data_path):
            args.data_path = os.path.join(QUEST_DIR, args.data_path)

        """set up sampler to datapreprocess"""
        train_sampler_class = import_class(f"unKR.data.{args.train_sampler_class}")
        train_sampler = train_sampler_class(args)

        test_sampler_class = import_class(f"unKR.data.{args.test_sampler_class}")
        test_sampler = test_sampler_class(train_sampler)

        """set up datamodule"""
        data_class = import_class(f"unKR.data.{args.data_class}")
        kgdata = data_class(args, train_sampler, test_sampler)

        """set up model"""
        model_class = import_class(f"unKR.model.{args.model_name}")
        model_meta = model_class(args)

        # === QUEST: inject graph data and (optional) spectral init ===
        if hasattr(model_meta, "set_graph_edges"):
            model_meta.set_graph_edges(
                train_sampler.train_triples, device=args.gpu
            )
            if getattr(model_meta, "use_spectral_init", True):
                model_meta.apply_spectral_init()
            else:
                print("  QUEST: spectral init disabled (Xavier init kept)")

        """set up lit_model"""
        litmodel_class = import_class(f"unKR.lit_model.{args.litmodel_name}")
        lit_model = litmodel_class(model_meta, args)
        print(lit_model)

        """set up logger"""
        logger = pl.loggers.TensorBoardLogger("training/logs")

        """early stopping — DISABLED so LP (WMRR) has time to converge.

        In the previous CN15k v1 run, MAE peaked at epoch 9 but WMRR was
        still climbing at epoch 109 when MAE-patience triggered stop.
        We now train for the full max_epochs and rely on ModelCheckpoint
        to save the best ckpt for each metric independently.
        """
        early_callback = pl.callbacks.EarlyStopping(
            monitor="Eval_wmrr",
            mode="max",
            patience=args.early_stop_patience * 5,  # 5x patience on WMRR
            check_on_train_epoch_end=False,
        )

        """set up model save method"""
        dirpath = "/".join(["output", args.eval_task, args.dataset_name,
                            args.model_name, ""])

        model_checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="Eval_MAE", mode="min",
            filename="{epoch}-{Eval_MAE:.5f}",
            dirpath=dirpath, save_weights_only=True, save_top_k=1,
        )
        model_checkpoint1 = pl.callbacks.ModelCheckpoint(
            monitor="Eval_wmrr", mode="max",
            filename="{epoch}-{Eval_wmrr:.5f}",
            dirpath=dirpath, save_weights_only=True, save_top_k=1,
        )
        model_checkpoint3 = pl.callbacks.ModelCheckpoint(
            monitor="Eval_MSE", mode="min",
            filename="{epoch}-{Eval_MSE:.5f}",
            dirpath=dirpath, save_weights_only=True, save_top_k=1,
        )
        model_checkpoint_last = pl.callbacks.ModelCheckpoint(
            filename="{epoch}", dirpath=dirpath,
            save_weights_only=True, save_top_k=-1,
            every_n_epochs=10, save_last=True,
        )
        callbacks = [early_callback, model_checkpoint, model_checkpoint1,
                     model_checkpoint3, model_checkpoint_last]

        trainer = pl.Trainer(
            callbacks=callbacks,
            logger=logger,
            default_root_dir="training/logs",
            accelerator="gpu",
            devices=[gpu_id],
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            max_epochs=args.max_epochs,
        )

        if args.save_config:
            save_config(args)

        if not args.test_only:
            trainer.fit(lit_model, datamodule=kgdata)
            # Test on BOTH best-MSE and best-WMRR checkpoints so we can
            # compare how each metric responds at the optimal point.
            paths = {
                "best-MSE": model_checkpoint3.best_model_path,
                "best-WMRR": model_checkpoint1.best_model_path,
            }
        else:
            paths = {"explicit": ""}

        for name, path in paths.items():
            if not path:
                continue
            print(f"\n=== Testing with {name} checkpoint: {path} ===",
                  flush=True)
            lit_model.load_state_dict(torch.load(path)["state_dict"])
            lit_model.eval()
            trainer.test(lit_model, datamodule=kgdata)


if __name__ == "__main__":
    main()
