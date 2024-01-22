"""
Working lightning Module
"""
from typing import List

import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn


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

        # TODO: other optimizer
        # scheduler.step()

        self.log(
            "train_loss", loss.mean().item(), prog_bar=True, on_step=True, on_epoch=True
        )
        # self.my_logger.debug(
        #     f"At batch {batch_idx} we are looking at ref_text {ref_text}  and triplets {ref_triplets}"
        # )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

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

        # 使用wandb.Image封裝圖片
        wandb_original = wandb.Image(original_image, caption="Original")
        wandb_reconstruction = wandb.Image(
            reconstructed_image, caption=f"{self.global_step}th g-step Reconstruction"
        )
        # 上傳圖片到wandb
        # report_dict["val_examples"] = [wandb_original, wandb_reconstruction]

        self.log(
            "val_loss", loss.mean().item(), prog_bar=True, on_step=True, on_epoch=True
        )
        wandb_logger = self.logger.experiment  # type:ignore
        if isinstance(wandb_logger, WandbLogger):
            wandb_logger.log(  # type:ignore
                "validation_images", [wandb_original, wandb_reconstruction]
            )

        return loss
