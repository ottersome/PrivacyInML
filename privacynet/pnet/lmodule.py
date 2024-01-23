import lightning as L
import torch


class TraningModule(L.LightningModule):
    def __init__(self):
        pass

    def training_step(self, batch):
        images, labels = batch

    def validation_step(self, batch):
        images, og_labels = batch

        # Generate Target Domain labels randomly
        rand_idx = torch.randperm(og_labels.size(1))  # CHECK: size
        trg_labels = og_labels[rand_idx]

        og_c = og_label.clone()
        trg_c = trg_labels.clone()
        zero = torch.zeros(images)
