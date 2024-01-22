"""
Working lightning Module
"""
from pathlib import Path
from typing import List

import cv2
import lightning as L
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger
from pml.utils import setup_logger  # type: ignore
from torch import Tensor, nn
from torch.optim.lr_scheduler import StepLR  # or any other scheduler

import wandb


class ReconstructionModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        # criterion as loss type
        criterion: nn.Module,
    ):
        """
        Arguments
        ---------
            parent_model_name: Name of model to use from HuggingFace's repertoire,
            lr: learning rate
            dtype: precision for model, we'll likely need float16 for smaller gpus
            useParentWeights: whether or not to load a checkpoint or just used weigths provided by 🤗

        """
        super().__init__()

        self.mylogger = setup_logger(
            __class__.__name__, log_path=Path(__file__).resolve().parent
        )
        self._recon_criterion = criterion
        # Create Configuration as per Parent
        ####################
        # Base Model
        ####################
        self.model = model

    def forward(self, batch, attention_mask):
        """
        For inference, also for guessing batch siz
        """
        # Check if it is training
        inputs = batch

        # Use decoder for inference
        if not self.training:
            pass  # TODO: If necessary
        else:
            pass  # TODO: write teacher-forcing method here

        return

    def training_step(self, batches, batch_idx):
        # Forward Pass
        report_dict = {}
        batch_imgs, batch_labels = batches

        # Forward Pass
        output = self.model(batch_imgs)

        # Loss
        loss = self._recon_criterion(output, batch_imgs)
        # TODO: penalty loss

        report_dict["batch_loss"] = loss.item()

        self.log(
            "train_loss", loss.mean().item(), prog_bar=True, on_step=True, on_epoch=True
        )
        # self.my_logger.debug(
        #     f"At batch {batch_idx} we are looking at ref_text {ref_text}  and triplets {ref_triplets}"
        # )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def validation_step(self, ref_batches: List[Tensor], batch_idx):
        # Whole of validation is here:
        batch_imgs, batch_labels = ref_batches

        output = self.model(batch_imgs)

        loss = self._recon_criterion(output, batch_imgs)
        # val_loss = (batch_idx * val_loss) / (batch_idx + 1) + (
        #     1 / (batch_idx + 1)
        # ) * loss.item()

        # 進行推理，獲得重建的圖片
        reconstructions = self.model(batch_imgs)

        # 選擇一個樣本來可視化
        original_image = batch_imgs[0]  # 假設inputs是一個batch的圖片
        reconstructed_image = reconstructions[0]
        # Equalize Reconstructed Image
        # reconstructed_image = reconstructed_image.squeeze().detach().cpu().numpy()
        # reconstructed_image = reconstructed_image * 255
        # reconstructed_image = reconstructed_image.astype(np.uint8)
        # reconstructed_image = cv2.equalizeHist(reconstructed_image)

        # 使用wandb.Image封裝圖片
        if batch_idx == 0:
            wandb_original = wandb.Image(original_image, caption="Original")
            wandb_reconstruction = wandb.Image(
                reconstructed_image,
                caption=f"{self.global_step}th g-step Reconstruction",
            )
            wandb_logger = self.logger.experiment  # type:ignore
            self.mylogger.info(f"wandb_logger is of type {type(wandb_logger)}")
            wandb_logger.log(  # type:ignore
                {"validation_images": [wandb_original, wandb_reconstruction]}
            )
        # 上傳圖片到wandb
        # report_dict["val_examples"] = [wandb_original, wandb_reconstruction]

        self.log(
            "val_loss", loss.mean().item(), prog_bar=True, on_step=True, on_epoch=True
        )

        return loss
