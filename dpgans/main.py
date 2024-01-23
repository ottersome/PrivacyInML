"""
Simple AutoEncoder
"""
# TODO:
# - [ ] Maybe collate

import json
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any, Dict, List

import debugpy
import lightning as L
import torch
import torch.nn.functional as F
import wandb
from ae.data import DataModule
from ae.models import SimpleAutoEncoder, UNet
from ae.modules import ReconstructionModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner.tuning import Tuner
from torch import nn
from torchvision import transforms
from tqdm import tqdm

parent_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_path))
from pml.utils import setup_logger  # type: ignore

torch.set_float32_matmul_precision("medium")


def af():
    ap = ArgumentParser()
    ap.add_argument("--epochs", default=5)
    ap.add_argument("--batch_size", default=32)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--cache_path", default="./.cache")
    ap.add_argument("--name_label_info", default="list_attr_celeba.txt")
    ap.add_argument("--selected_attrs", type=List, default=["Male", "Eyeglasses"])
    ap.add_argument("--encoder_dim", type=int, default=128)

    ap.add_argument("--latent_dim", type=int, default=1024)
    ap.add_argument("--log_interval", type=int, default=5)
    ap.add_argument("--recon_lr", type=int, default=1e-3)
    ap.add_argument(
        "--eval_period", type=int, default=500, help="How many epochs before eval"
    )
    ap.add_argument("--split_percents", default=[0.8, 0.1, 0.1])
    ap.add_argument("-d", "--debug", action="store_true")
    ap.add_argument("-p", "--port", default=42018)
    ap.add_argument("--json_structure", default="./model_specs/unet0.json")

    ap.add_argument("-w", "--wandb", action="store_true")
    ap.add_argument("--wpname", default="PrivateAutoEncoder", help="WANDB Project Name")
    ap.add_argument("--wrname", default=None, help="WANDB run name")
    ap.add_argument("--wrnotes", default=None, help="WANDB run notes")
    ap.add_argument("--wrtags", default=[], help="WANDB run tags")

    args = ap.parse_args()
    args.cache_path = Path(args.cache_path).resolve()
    return args


args = af()
logger = setup_logger("MAIN", log_path=Path(__file__).resolve().parent)

if args.debug:
    logger.info("Waiting for debugpy client to attach")
    debugpy.listen(("0.0.0.0", args.port))
    debugpy.wait_for_client()
    logger.info("Client connected. Resuming with debugging session.")


L.seed_everything(0)
datamodule = DataModule(
    root=args.data_dir,
    attr_path=os.path.join(args.data_dir, args.name_label_info),
    cache_path=args.cache_path,
    selected_attrs=args.selected_attrs,
    transform=transforms.ToTensor(),
    batch_size=args.batch_size,
    split_percents=args.split_percents,
)
datamodule.prepare_data()

recon_criterion = nn.BCEWithLogitsLoss()
# recon_criterion = nn.CrossEntropyLoss()
# sensitive_penalty = nn.MSELoss()  # TODO:  set the right criterion
args.criterion = recon_criterion.__class__.__name__


# Get AutoEncoder
json_structure = []
with open(args.json_structure, "r") as f:
    json_structure = json.load(f)

# Get Image Data sizes from dataloader/dataset
logger.info(f"Running Model with struture {json.dumps(json_structure, indent=4)}")
# model = UNet(
#     json_structure,
#     {
#         "channels": datamodule.channels,
#         "height": datamodule.image_height,
#         "width": datamodule.image_width,
#     },
# )
model = SimpleAutoEncoder()
# gender_classifier =
# Optimizer
recon_optimizer = torch.optim.Adam(model.parameters(), lr=args.recon_lr)
# penalty_optimizer = torch.optim.Adam(model.parameters())
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
wandb_logger = None
if args.wandb:
    wandb_logger = WandbLogger(
        project=args.wpname,
        name=args.wrname,
        notes=args.wrnotes,
        tags=args.wrtags,
        config=vars(args),
    )

lightning_module = ReconstructionModule(model, recon_criterion)


checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints", save_top_k=3, monitor="val_loss"
)

trainer = L.Trainer(
    accelerator="gpu",
    logger=wandb_logger,
    max_epochs=args.epochs,
    val_check_interval=0.125,
    log_every_n_steps=1,
    enable_checkpointing=True,
    callbacks=[checkpoint_callback],
)

logger.info("Using Tuner to find batch size")
tuner = Tuner(trainer)
# tuner.scale_batch_size(lightning_module, mode="binsearch", datamodule=datamodule)


logger.info("Fitting Model")
trainer.fit(lightning_module, datamodule=datamodule)  # type: ignore

logger.info("Saving checkpoint")
trainer.save_checkpoint(args.checkpoint_path)
