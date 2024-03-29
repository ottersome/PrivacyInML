import datetime
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torchvision.utils import save_image
from tqdm import tqdm

from deepface import DeepFace
from pnet.losses import DescriptorLoss
from pnet.models import Discriminator, Generator, VGGFace

cur_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(cur_path))
from pml.utils import create_logger  # type:ignore


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Logging
        self.locallogger = create_logger(__class__.__name__)
        self.use_wandb = config.wandb

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader

        # Run Config
        self.run_name = config.wrname
        self.run_notes = config.wrnotes

        # Cheaty Config
        self.cconfig = config

        # Model configurations.
        self.c_dim = config.c_dim  # Dimension of labels (First Dataset)
        self.c2_dim = config.c2_dim  # Dimensions of labels (Second Dataset)
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_ds = config.lambda_ds
        self.lambda_gp = config.lambda_gp
        self.facedesc_weights_loc = config.facedesc_weights_loc

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        self.locallogger.info(f"Using data located at {self.dataset}")

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Identifiability(Downstream) Loss
        self.ident_loss = config.identifiability

        # Ensure existance of save_dir
        if os.path.exists(self.model_save_dir) == False:
            os.makedirs(self.model_save_dir)

        # Create Sample dir if not exists
        if os.path.exists(self.sample_dir) == False:
            os.makedirs(self.sample_dir)

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model(config.ptrnd_D, config.ptrnd_G)
        if self.use_wandb:
            self.build_remotelogger()

    def build_model(self, ptrnd_D: str, ptrnd_G: str):
        """Create a generator and a discriminator."""

        if self.dataset in ["CelebA", "RaFD"]:
            amnt_attrs = len(self.selected_attrs)
            self.G = Generator(amnt_attrs)
            self.D = Discriminator(
                self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num
            )
            # Load Weights
            both_GD_exist = [
                os.path.exists(ptrnd_D),
                os.path.exists(ptrnd_G),
            ]
            if all(both_GD_exist):
                self.locallogger.info(
                    "💾 Loading pretrained weights:"
                    f"\n\tDiscriminator {ptrnd_D}"
                    f"\n\tDiscriminator {ptrnd_G}"
                )
                self.G.load_state_dict(torch.load(ptrnd_G))
                self.D.load_state_dict(torch.load(ptrnd_D))

        elif self.dataset in ["Both"]:
            assert False, "You should not be here"
            amnt_attrs = len(self.celeba_loader.dataset.attr2idx)
            self.G = Generator(amnt_attrs)
            self.D = Discriminator(
                self.image_size,
                self.d_conv_dim,
                self.c_dim + self.c2_dim,
                self.d_repeat_num,
            )

        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), self.g_lr, [self.beta1, self.beta2]  # type: ignore
        )
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), self.d_lr, [self.beta1, self.beta2]  # type: ignore
        )
        self.print_network(self.G, "G")
        self.print_network(self.D, "D")

        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print("Loading the trained models from step {}...".format(resume_iters))
        G_path = os.path.join(self.model_save_dir, "{}-G.ckpt".format(resume_iters))
        D_path = os.path.join(self.model_save_dir, "{}-D.ckpt".format(resume_iters))
        self.G.load_state_dict(
            torch.load(G_path, map_location=lambda storage, loc: storage)
        )
        self.D.load_state_dict(
            torch.load(D_path, map_location=lambda storage, loc: storage)
        )

    def build_remotelogger(self):
        """Build a tensorboard logger."""
        wandblogger = None
        run_name = self.run_name if self.run_name != "" else None
        run_notes = self.run_notes if self.run_notes != "" else None
        configs = vars(self.cconfig)

        if self.use_wandb == True:
            wandblogger = wandb.init(
                project="StarGAN", name=run_name, notes=run_notes, config=configs  # type: ignore
            )

        self.logger = wandblogger

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group["lr"] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group["lr"] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset="CelebA", selected_attrs=None):
        """
        Generate target domain labels for debugging and testing.
        Each row will contain a single bit-flip from original labels
        """
        # Get hair color indices.
        if dataset == "CelebA":
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == "CelebA":
                c_trg = c_org.clone()
                if (  # type: ignore
                    i in hair_color_indices  # type:ignore
                ):  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:  # type: ignore
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = c_trg[:, i] == 0  # Reverse attribute value.
            elif dataset == "RaFD":
                c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

            c_trg_list.append(c_trg.to(self.device))  # type: ignore
        return c_trg_list

    def classification_loss(self, logit, target, dataset="CelebA"):
        """Compute binary or softmax cross entropy loss."""
        if dataset == "CelebA":
            return F.binary_cross_entropy_with_logits(
                logit, target, size_average=False
            ) / logit.size(0)
        elif dataset == "RaFD":
            return F.cross_entropy(logit, target)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == "CelebA":
            data_loader = self.celeba_loader
        elif self.dataset == "RaFD":
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)  # type: ignore
        # x_fixed is for later human evaluation
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)[:4]
        c_fixed_list = self.create_labels(
            c_org[:4], self.c_dim, self.dataset, self.selected_attrs
        )

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print("Start training...")
        start_time = time.time()
        iterbar = tqdm(range(start_iters, self.num_iters), desc="Training")
        for i in range(start_iters, self.num_iters):
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)  # type: ignore
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == "CelebA":
                # CelebA are already one hot encoded
                c_org = label_org.clone()
                c_trg = label_trg.clone()

            elif self.dataset == "RaFD":
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)  # Input images.
            c_org = c_org.to(self.device)  # Original domain labels.
            c_trg = c_trg.to(self.device)  # Target domain labels.
            label_org = label_org.to(
                self.device
            )  # Labels for computing classification loss.
            label_trg = label_trg.to(
                self.device
            )  # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = -torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(
                True
            )
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = (
                d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
            ) + self.lambda_cls * d_loss_cls  # AdvLoss  # Classificaition Loss
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss["D/loss_real"] = d_loss_real.item()
            loss["D/loss_fake"] = d_loss_fake.item()
            loss["D/loss_cls"] = d_loss_cls.item()
            loss["D/loss_gp"] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = -torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # TODO: the Descriptor loss
                g_loss_ident = 0
                ident_losses = []
                if self.ident_loss:
                    for b in range(x_real.shape[0]):
                        ocv_real = x_real[b].permute(1, 2, 0)
                        ocv_fake = x_fake[b].permute(1, 2, 0)
                        real_embed = torch.Tensor(
                            DeepFace.represent(
                                ocv_real.detach().cpu().numpy(),
                                enforce_detection=False,
                            )[0]["embedding"]
                        )
                        fake_embed = torch.Tensor(
                            DeepFace.represent(
                                ocv_fake.detach().cpu().numpy(), enforce_detection=False
                            )[0]["embedding"]
                        )
                        # Calcualte the eucledian distance
                        euc_dist = torch.norm(real_embed - fake_embed, p=2)
                        cosine_dist = F.cosine_similarity(
                            real_embed.unsqueeze(0), fake_embed.unsqueeze(0)
                        )
                        if (i + 1) % self.log_step == 0:
                            self.locallogger.debug(
                                f"The euc_dist is {euc_dist} while cosine_dist {cosine_dist}"
                            )
                        ident_losses.append(cosine_dist)
                    g_loss_ident = torch.stack(ident_losses).mean()

                # Backward and optimize.
                g_loss = (
                    g_loss_fake
                    + self.lambda_rec * g_loss_rec
                    + self.lambda_cls * g_loss_cls
                    + self.lambda_ds * g_loss_ident
                )

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss["G/loss_fake"] = g_loss_fake.item()
                loss["G/loss_rec"] = g_loss_rec.item()
                loss["G/loss_cls"] = g_loss_cls.item()
                if self.ident_loss:
                    loss["G/loss_ident"] = g_loss_ident.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(
                    et, i + 1, self.num_iters
                )
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_wandb:
                    for tag, value in loss.items():
                        # self.logger.scalar_summary(tag, value, i + 1)
                        self.logger.log({tag: value})

            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                x_fake_list = [x_fixed]
                for c_fixed in c_fixed_list:
                    x_fake_list.append(self.G(x_fixed, c_fixed))
                x_concat = torch.cat(x_fake_list, dim=3)
                sample_path = os.path.join(
                    self.sample_dir, "{}-images.jpg".format(i + 1)
                )
                save_image(
                    self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0
                )
                # Send Image to Wandb
                if self.use_wandb:
                    self.logger.log({"sample": [wandb.Image(sample_path)]})
                print("Saved real and fake images into {}...".format(sample_path))

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(
                    self.model_save_dir, "{}-G_baseline.ckpt".format(i + 1)
                )
                D_path = os.path.join(
                    self.model_save_dir, "{}-D_baseline.ckpt".format(i + 1)
                )
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print("Saved model checkpoints into {}...".format(self.model_save_dir))

            # Linear Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (
                self.num_iters - self.num_iters_decay
            ):
                g_lr -= self.g_lr / float(self.num_iters_decay)
                d_lr -= self.d_lr / float(self.num_iters_decay)
                self.update_lr(g_lr, d_lr)
                print("Decayed learning rates, g_lr: {}, d_lr: {}.".format(g_lr, d_lr))

            iterbar.update(1)

    def save_labeled_image(
        self, x_fixed: torch.Tensor, c_fixed: torch.Tensor, attr_list: List[str]
    ):
        """
        Take Fixed Image for debuggings and their labels
        and plot them in a nice plot with labels
        """
        with torch.no_grad():
            x_fake_list = [x_fixed]
            for c_fixed in c_fixed_list[:4]:
                x_fake_list.append(self.G(x_fixed, c_fixed))
            x_concat = torch.cat(x_fake_list, dim=3)
            sample_path = os.path.join(self.sample_dir, "{}-images.jpg".format(i + 1))
            save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
            print("Saved real and fake images into {}...".format(sample_path))

    def train_multi(self):
        """Train StarGAN with multiple datasets."""
        # Data iterators.
        celeba_iter = iter(self.celeba_loader)
        rafd_iter = iter(self.rafd_loader)

        # Fetch fixed inputs for debugging.
        x_fixed, c_org = next(celeba_iter)
        x_fixed = x_fixed.to(self.device)
        c_celeba_list = self.create_labels(
            c_org, self.c_dim, "CelebA", self.selected_attrs
        )
        c_rafd_list = self.create_labels(c_org, self.c2_dim, "RaFD")
        zero_celeba = torch.zeros(x_fixed.size(0), self.c_dim).to(
            self.device
        )  # Zero vector for CelebA.
        zero_rafd = torch.zeros(x_fixed.size(0), self.c2_dim).to(
            self.device
        )  # Zero vector for RaFD.
        mask_celeba = self.label2onehot(torch.zeros(x_fixed.size(0)), 2).to(
            self.device
        )  # Mask vector: [1, 0].
        mask_rafd = self.label2onehot(torch.ones(x_fixed.size(0)), 2).to(
            self.device
        )  # Mask vector: [0, 1].

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print("Start training...")
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            data_iter = celeba_iter

            try:
                x_real, label_org = next(data_iter)
            except:
                if dataset == "CelebA":
                    celeba_iter = iter(self.celeba_loader)
                    x_real, label_org = next(celeba_iter)
                elif dataset == "RaFD":
                    rafd_iter = iter(self.rafd_loader)
                    x_real, label_org = next(rafd_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if dataset == "CelebA":
                c_org = label_org.clone()
                c_trg = label_trg.clone()
                zero = torch.zeros(x_real.size(0), self.c2_dim)
                mask = self.label2onehot(torch.zeros(x_real.size(0)), 2)
                c_org = torch.cat([c_org, zero, mask], dim=1)
                c_trg = torch.cat([c_trg, zero, mask], dim=1)
            elif dataset == "RaFD":
                c_org = self.label2onehot(label_org, self.c2_dim)
                c_trg = self.label2onehot(label_trg, self.c2_dim)
                zero = torch.zeros(x_real.size(0), self.c_dim)
                mask = self.label2onehot(torch.ones(x_real.size(0)), 2)
                c_org = torch.cat([zero, c_org, mask], dim=1)
                c_trg = torch.cat([zero, c_trg, mask], dim=1)

            x_real = x_real.to(self.device)  # Input images.
            c_org = c_org.to(self.device)  # Original domain labels.
            c_trg = c_trg.to(self.device)  # Target domain labels.
            label_org = label_org.to(
                self.device
            )  # Labels for computing classification loss.
            label_trg = label_trg.to(
                self.device
            )  # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            out_cls = (
                out_cls[:, : self.c_dim]
                if dataset == "CelebA"
                else out_cls[:, self.c_dim :]
            )
            d_loss_real = -torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, dataset)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, _ = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(
                True
            )
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = (
                d_loss_real
                + d_loss_fake
                + self.lambda_cls * d_loss_cls
                + self.lambda_gp * d_loss_gp
            )
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss["D/loss_real"] = d_loss_real.item()
            loss["D/loss_fake"] = d_loss_fake.item()
            loss["D/loss_cls"] = d_loss_cls.item()
            loss["D/loss_gp"] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                out_cls = (
                    out_cls[:, : self.c_dim]
                    if dataset == "CelebA"
                    else out_cls[:, self.c_dim :]
                )
                g_loss_fake = -torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, dataset)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = (
                    g_loss_fake
                    + self.lambda_rec * g_loss_rec
                    + self.lambda_cls * g_loss_cls
                )
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss["G/loss_fake"] = g_loss_fake.item()
                loss["G/loss_rec"] = g_loss_rec.item()
                loss["G/loss_cls"] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training info.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(
                    et, i + 1, self.num_iters, dataset
                )
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_wandb:
                    for tag, value in loss.items():
                        # self.logger.scalar_summary(tag, value, i + 1)
                        self.logger.log({tag: value})

            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_celeba_list:
                        c_trg = torch.cat([c_fixed, zero_rafd, mask_celeba], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    for c_fixed in c_rafd_list:
                        c_trg = torch.cat([zero_celeba, c_fixed, mask_rafd], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(
                        self.sample_dir, "{}-images.jpg".format(i + 1)
                    )
                    save_image(
                        self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0
                    )
                    print("Saved real and fake images into {}...".format(sample_path))

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, "{}-G.ckpt".format(i + 1))
                D_path = os.path.join(self.model_save_dir, "{}-D.ckpt".format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print("Saved model checkpoints into {}...".format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (
                self.num_iters - self.num_iters_decay
            ):
                g_lr -= self.g_lr / float(self.num_iters_decay)
                d_lr -= self.d_lr / float(self.num_iters_decay)
                self.update_lr(g_lr, d_lr)
                print("Decayed learning rates, g_lr: {}, d_lr: {}.".format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        if self.dataset == "CelebA":
            data_loader = self.celeba_loader
        elif self.dataset == "RaFD":
            data_loader = self.rafd_loader

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):
                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(
                    c_org, self.c_dim, self.dataset, self.selected_attrs
                )

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(
                    self.result_dir, "{}-images.jpg".format(i + 1)
                )
                save_image(
                    self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0
                )
                print("Saved real and fake images into {}...".format(result_path))

    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(self.celeba_loader):
                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_celeba_list = self.create_labels(
                    c_org, self.c_dim, "CelebA", self.selected_attrs
                )
                c_rafd_list = self.create_labels(c_org, self.c2_dim, "RaFD")
                zero_celeba = torch.zeros(x_real.size(0), self.c_dim).to(
                    self.device
                )  # Zero vector for CelebA.
                zero_rafd = torch.zeros(x_real.size(0), self.c2_dim).to(
                    self.device
                )  # Zero vector for RaFD.
                mask_celeba = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(
                    self.device
                )  # Mask vector: [1, 0].
                mask_rafd = self.label2onehot(torch.ones(x_real.size(0)), 2).to(
                    self.device
                )  # Mask vector: [0, 1].

                # Translate images.
                x_fake_list = [x_real]
                for c_celeba in c_celeba_list:
                    c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))
                for c_rafd in c_rafd_list:
                    c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(
                    self.result_dir, "{}-images.jpg".format(i + 1)
                )
                save_image(
                    self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0
                )
                print("Saved real and fake images into {}...".format(result_path))
