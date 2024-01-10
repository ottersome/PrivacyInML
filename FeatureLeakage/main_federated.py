from argparse import ArgumentParser
from logging import DEBUG
from pathlib import Path
from typing import Dict, List

import debugpy
import pandas as pd
import torch
import torch.multiprocessing as mp
from debugpy import breakpoint
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import UTKDataset, utk_parse_federated
from model import VGGRegressionModel
from utils import create_logger

main_logger = create_logger("main", DEBUG)


def apfun():
    ap = ArgumentParser()
    # ap.add_argument("--prototypes", default=10, type=int, help="Amount of Prototypes")
    ap.add_argument(
        "--data_path", default="./data/utkface", type=str, help="Path to dataset"
    )
    ap.add_argument(
        "--batch_size", default=32, type=int, help="Batch Size for Training"
    )
    ap.add_argument("-d", "--debugpy", help="Attach debugpy", action="store_true")
    ap.add_argument("-p", "--port", default=42019, help="Port for debugpy")
    ap.add_argument("--lr", default=1e-3, type=float, help="Learning Rate")
    ap.add_argument("--epochs", default=200, type=int, help="Amount of Epochs")
    ap.add_argument(
        "--epoch_of_seggregation",
        default=101,
        type=int,
        help="Batch at which new data is inserted",
    )
    ap.add_argument(
        "--batch_of_seggregation",
        default=20,
        type=int,
        help="Batch at which new data is inserted",
    )
    ap.add_argument("--segg_race", default=1)
    ap.add_argument("--train_proportion", default=0.8)
    ap.add_argument(
        "--warmpup_epochs", default=12, help="Epochs before introducing change"
    )

    args = ap.parse_args()
    assert (
        args.warmpup_epochs < args.batch_of_seggregation
    ), "Seggregation must occur after warmpup_epochs"
    # Sanitize

    # TODO: maybe add dataset source
    return args


def create_clients_DataLoaders(train: List[pd.DataFrame], batch_size: int):
    cdls = []
    for df in train:
        img_data: List = df["img_data"].values.tolist()
        age_data: List = df["ages"].values.tolist()
        race_data: List = df["races"].values.tolist()

        ds = UTKDataset(img_data, age_data, race_data)
        cdls.append(DataLoader(ds, batch_size))
    return cdls


def federated_orchestrator(train_dl: List[torch.utils.data.DataLoader], args):
    manager = mp.Manager()
    processes = []
    client_models = [VGGRegressionModel().to(f"cuda:{i}") for i in range(len(train_dl))]
    barrier = mp.Barrier(len(client_models))
    for cm in client_models:
        cm.share_memory()

    shared_weights = manager.list(
        [
            {k: v.cpu().numpy() for k, v in model.state_dict().items()}
            for model in client_models
        ]
    )
    optimizers = [torch.optim.Adam(model.parameters()) for model in client_models]
    criterion = torch.nn.MSELoss()

    for i, dataloader in enumerate(train_dl):
        model = client_models[i]
        main_logger.info(f"Creating Process for the {i}th client")
        p = mp.Process(
            target=per_client_trains,
            args=(
                i,
                dataloader,
                model,
                criterion,
                optimizers[i],
                args.epochs,
                shared_weights[i],
                barrier,
            ),
        )
        main_logger.info(f"Starting Process for the {i}th client")
        p.start()
        main_logger.info(f"Appending Process for the {i}th client")
        processes.append(p)

    # Wait for them to finish
    main_logger.info(f"Joining Processes")
    for p in processes:
        p.join()


def avg_weights(weights: List[Dict]):
    base = {}
    for k, _ in weights[0].items():
        base[k] = sum([w[k].cpu().numpy() for w in weights]) / len(weights)
    return base


def per_client_trains(
    idx: int,
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: nn.Module,
    epochs,
    shared_weights,
    barrier,
):
    main_logger.debug(f"Client {idx} has entered the chat.")
    torch.cuda.set_device(idx)

    for _ in range(epochs):
        for batchx, batchy in dataloader:
            optimizer.zero_grad()
            pred = model(batchx)
            loss = criterion(pred, batchy)
            loss.backward()

            with shared_weights.get_lock():
                shared_weights[idx] = {
                    k: v.cpu().numpy() for k, v in model.state_dict.items()
                }

            # Wait for all to compute their new weights locally
            barrier.wait()

            if idx == 0:  # Let idx 0 be coordinator
                with shared_weights.get_lock():
                    avg = avg_weights(shared_weights.values())
                    for i in range(len(shared_weights)):
                        shared_weights[i] = avg

            # Weight for all of them to be upadted
            barrier.wait()
            model.load_state_dict(
                {
                    k: torch.Tensor(v).to(f"cuda:{idx}")
                    for k, v in shared_weights[idx].items()
                }
            )


if __name__ == "__main__":
    args = apfun()
    main_logger.info("Starting script.")
    mp.set_start_method("spawn")

    if args.debugpy:
        conn_tuple = ("0.0.0.0", 42019)
        main_logger.info("Waiting for debugpy attachment on ")
        debugpy.listen(conn_tuple)
        debugpy.wait_for_client()

    num_gpus = torch.cuda.device_count()

    # Load The Data
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Path {data_path} does not exist")
    main_logger.info("Getting data")
    trains, test, segg_data = utk_parse_federated(
        path=data_path,
        batch_size=args.batch_size,
        num_clients=num_gpus,
        batch_of_insertion=args.batch_of_seggregation,
        tt_split=0.8,
        race_of_interest=args.segg_race,
    )

    dataloaders = create_clients_DataLoaders(trains, args.batch_size)

    # Train the models
    federated_orchestrator(dataloaders, args)
