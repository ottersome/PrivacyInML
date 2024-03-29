"""
Derivative of 
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

Their work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import argparse
import os
import sys
from pathlib import Path

parent_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_path))

import logging

import debugpy
import torch
from munch import Munch
from pml.utils import setup_logger  # type: ignore
from torch.backends import cudnn

from pnet.confounding_solver import Solver
from pnet.data import get_loader


def str2bool(v):
    return v.lower() in ("true")


def subdirs(dname):
    return [d for d in os.listdir(dname) if os.path.isdir(os.path.join(dname, d))]


def main(config: argparse.Namespace):
    cudnn.benchmark = True
    # torch.manual_seed(config.seed)

    celeba_loader = get_loader(
        config.celeba_image_dir,
        config.attr_path,
        config.selected_attrs,
        config.celeba_crop_size,
        config.image_size,
        config.batch_size,
        "CelebA",
        config.mode,
        config.num_workers,
    )
    cur_loc = Path(__file__).resolve().parent / "./logs"
    if not cur_loc.exists():
        cur_loc.mkdir()

    main_logger = setup_logger("MAIN", logging.INFO, cur_loc)

    if config.debug:
        main_logger.info("Waiting for debugpy client to attach")
        debugpy.listen(("0.0.0.0", args.port))
        debugpy.wait_for_client()
        main_logger.info("Client connected. Resuming with debugging session.")

    solver = Solver(celeba_loader, None, config)

    if config.mode == "train":
        if config.dataset in ["CelebA", "RaFD"]:
            solver.train()
        elif config.dataset in ["Both"]:
            solver.train_multi()
    elif config.mode == "test":
        if config.dataset in ["CelebA", "RaFD"]:
            solver.test()
        elif config.dataset in ["Both"]:
            solver.test_multi()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Use debugpy to debug",
    )
    parser.add_argument(
        "-w",
        "--wandb",
        action="store_true",
        default=False,
        help="Use debugpy to debug",
    )
    parser.add_argument("--wrname", default="", type=str, help="WanDBAI Run name")
    parser.add_argument("--wrnotes", default="", type=str, help="WanDBAI Run Notes")

    parser.add_argument("-p", "--port", default=42019, type=int)
    parser.add_argument(
        "--c_dim", type=int, default=5, help="dimension of domain labels (1st dataset)"
    )
    parser.add_argument(
        "--c2_dim", type=int, default=8, help="dimension of domain labels (2nd dataset)"
    )
    parser.add_argument(
        "--celeba_crop_size",
        type=int,
        default=178,
        help="crop size for the CelebA dataset",
    )
    parser.add_argument(
        "--rafd_crop_size", type=int, default=256, help="crop size for the RaFD dataset"
    )
    parser.add_argument("--image_size", type=int, default=128, help="image resolution")
    parser.add_argument(
        "--g_conv_dim",
        type=int,
        default=64,
        help="number of conv filters in the first layer of G",
    )
    parser.add_argument(
        "--d_conv_dim",
        type=int,
        default=64,
        help="number of conv filters in the first layer of D",
    )
    parser.add_argument(
        "--g_repeat_num", type=int, default=6, help="number of residual blocks in G"
    )
    parser.add_argument(
        "--d_repeat_num", type=int, default=6, help="number of strided conv layers in D"
    )
    parser.add_argument(
        "--lambda_cls",
        type=float,
        default=5,
        help="weight for domain classification loss",
    )
    parser.add_argument(
        "--lambda_rec", type=float, default=10, help="weight for reconstruction loss"
    )
    parser.add_argument(
        "--lambda_gp", type=float, default=10, help="weight for gradient penalty"
    )
    parser.add_argument("--lambda_ds", default=10, help="weight for Identifiability")
    parser.add_argument("--lambda_cfd", default=1, help="weight for confounding")

    # Training configuration.
    parser.add_argument(
        "--dataset", type=str, default="CelebA", choices=["CelebA", "RaFD", "Both"]
    )
    parser.add_argument("--batch_size", type=int, default=64, help="mini-batch size")
    parser.add_argument(
        "--num_iters",
        type=int,
        default=200000,
        help="number of total iterations for training D",
    )
    parser.add_argument(
        "--num_iters_decay",
        type=int,
        default=100000,
        help="number of iterations for decaying lr",
    )
    parser.add_argument(
        "--g_lr", type=float, default=0.0001, help="learning rate for G"
    )
    parser.add_argument(
        "--d_lr", type=float, default=0.0001, help="learning rate for D"
    )
    parser.add_argument(
        "--n_critic", type=int, default=5, help="number of D updates per each G update"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="beta2 for Adam optimizer"
    )
    parser.add_argument(
        "--resume_iters", type=int, default=None, help="resume training from this step"
    )
    parser.add_argument(
        "--selected_attrs",
        nargs="+",
        help="selected attributes for the CelebA dataset",
        default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
    )
    parser.add_argument(
        "--cfd_attrs",
        nargs="+",
        help="selected attributes for the CelebA dataset obfuscation.",
        default=["Male"],
    )

    # Test configuration.
    parser.add_argument(
        "--test_iters", type=int, default=200000, help="test model from this step"
    )

    # Pretrained Weights
    parser.add_argument(
        "--facedesc_weights_loc", type=str, default="data/facedesc_weights.pth"
    )

    # Miscellaneous.
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--use_tensorboard", type=str2bool, default=True)

    # Directories.
    parser.add_argument("--celeba_image_dir", type=str, default="../data/imgs")
    parser.add_argument("--attr_path", type=str, default="../data/list_attr_celeba.txt")
    parser.add_argument("--rafd_image_dir", type=str, default="data/RaFD/train")
    parser.add_argument("--log_dir", type=str, default="stargan/logs")
    parser.add_argument("--model_save_dir", type=str, default="stargan/models")
    parser.add_argument("--sample_dir", type=str, default="stargan/samples")
    parser.add_argument("--result_dir", type=str, default="stargan/results")

    # Pretrained Weights
    parser.add_argument("--ptrnd_D", type=str, default="privategan/models/ptrnd_D.ckpt")
    parser.add_argument("--ptrnd_G", type=str, default="privategan/models/ptrnd_G.ckpt")

    # Step size.
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--sample_step", type=int, default=10)
    parser.add_argument("--model_save_step", type=int, default=10000)
    parser.add_argument("--lr_update_step", type=int, default=1000)
    parser.add_argument(
        "--description_gen", type=str, default="CelebA", choices=["CelebA"]
    )

    # Identifiability (Downstream)
    parser.add_argument("--identifiability", action="store_true", default=False)

    args = parser.parse_args()

    main(args)
