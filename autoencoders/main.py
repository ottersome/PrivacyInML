"""
Simple AutoEncoder
"""
# TODO:
# - [ ] Maybe collate

import os
from argparse import ArgumentParser
from typing import List

import torch
import wandb
from torch import nn
from tqdm import tqdm

from ae.data import CelebADataLoader, CelebADataset, Mode
from ae.models import ConvVAE

from ..pml.utils import setup_logger


def af():
    ap = ArgumentParser()
    ap.add_argument("--epochs", default=10)
    ap.add_argument("--batch_size", default=4)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--cache_path", default="./.cache")
    ap.add_argument("--name_label_info", default="list_attr_celeba.txt")
    ap.add_argument("--selected_attrs", type=List, default=["Male", "Eyeglasses"])
    ap.add_argument("--encoder_dim", type=int, default=128)
    ap.add_argument("--latent_dim", type=int, default=20)
    ap.add_argument("--log_interval", type=int, default=1)
    ap.add_argument(
        "--eval_period", type=int, default=10, help="How many epochs before eval"
    )
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--split_percents", default=[0.8, 0.1, 0.1])
    return ap.parse_args()


logger = setup_logger("MAIN", log_path=__file__)

args = af()
transform = lambda a: a

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():  # type:ignore
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Initialize Wandb
if args.wandb:
    logger.info("🪄 Instantiating WandB")
    wandb.init(project="PrivateAutoEncoder")
else:
    logger.warn("⚠️ Not using Wandb")


dataset = CelebADataset(
    args.data_dir,
    os.path.join(args.data_dir, args.name_label_info),
    args.cache_path,
    args.selected_attrs,
    transform,
    Mode.TRAIN,
    split_percents=args.split_percents,
)
dataloader = CelebADataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=1  # , drop_last=True
)

recon_criterion = nn.MSELoss()
sensitive_penalty = nn.MSELoss()  # TODO:  set the right criterion

# Get AutoEncoder
model = ConvVAE(dataset.image_dim, args.encoder_dim, args.latent_dim).to(device)
# Optimizer
recon_optimizer = torch.optim.Adam(model.parameters())
penalty_optimizer = torch.optim.Adam(model.parameters())
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training Loop
ebar = tqdm(range(args.epochs), desc="Epoch")
# bbar = tqdm(range(args.epochs), desc="Batch")
epoch_loss = 0
for epoch in range(args.epochs):
    report_dict = {}
    ebar.set_description(f"Last loss = {epoch_loss}, Going Through Batches")
    epoch_loss = 0
    for batch_idx, (batch_imgs, batch_labels) in enumerate(dataloader):
        # Forward Pass
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        # Zero the Gradients
        recon_optimizer.zero_grad()

        # Forward Pass
        output = model(batch_imgs)

        # Loss
        loss = recon_criterion(output, batch_imgs)
        # TODO: penalty loss

        # Backprop
        loss.backward()
        epoch_loss = (batch_idx * epoch_loss) / (batch_idx + 1) + (
            1 / (batch_idx + 1)
        ) * loss.item()

        recon_optimizer.step()
        # TODO: other optimizer
        # scheduler.step()

    # DONE BATCH

    # TODO: Change to recon loss
    report_dict["epoch_loss"] = epoch_loss

    ebar.set_description(f"Last loss = {epoch_loss}, Validating.")
    if epoch % args.eval_period == 0:
        model.eval()
        dataset.set_mode(Mode.VALIDATE)
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (batch_imgs, batch_labels) in enumerate(dataloader):
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)

                output = model(batch_imgs)
                loss = recon_criterion(output, batch_imgs)
                val_loss = (batch_idx * val_loss) / (batch_idx + 1) + (
                    1 / (batch_idx + 1)
                ) * loss.item()

        report_dict["val_loss"] = val_loss

        # Single sample from datalaoder
        sample = next(iter(dataloader))

        # 進行推理，獲得重建的圖片
        reconstructions = model(sample)

        # 選擇一個樣本來可視化
        original_image = sample[0]  # 假設inputs是一個batch的圖片
        reconstructed_image = reconstructions[0]

        # 使用wandb.Image封裝圖片
        wandb_original = wandb.Image(original_image, caption="Original")
        wandb_reconstruction = wandb.Image(
            reconstructed_image, caption=f"{epoch}th Epoch Reconstruction"
        )
        report_dict["val_example"][wandb_original, wandb_reconstruction]

        # 上傳圖片到wandb
        wandb.log({"examples": [wandb_original, wandb_reconstruction]})

        # 可能只想上傳一個batch的圖片，所以可以break
        break

    wandb.log(report_dict)
    ebar.update(1)
    ebar.set_description(f"")
