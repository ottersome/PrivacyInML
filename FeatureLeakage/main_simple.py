from argparse import ArgumentParser
from math import ceil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import UTKDataset, utk_parse
from model import VGGRegressionModel


def apfun():
    ap = ArgumentParser()
    # ap.add_argument("--prototypes", default=10, type=int, help="Amount of Prototypes")
    ap.add_argument(
        "--data_path", default="./data/utkface", type=str, help="Path to dataset"
    )
    ap.add_argument(
        "--batch_size", default=32, type=int, help="Batch Size for Training"
    )
    ap.add_argument("--epochs", default=100, type=int, help="Amount of Epochs")
    ap.add_argument("--train_proportion", default=0.8)

    # TODO: maybe add dataset source
    return ap.parse_args()


if __name__ == "__main__":
    args = apfun()

    # Load The Data
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Path {data_path} does not exist")
    imgs, ages, genders, races = utk_parse(data_path)

    ts = int(args.train_proportion * len(imgs))
    imgs_train, imgs_test = imgs[:ts], imgs[ts:]
    labels_train, labels_test = (ages[:ts], ages[ts:])

    # Dataset and DataLoader
    ds_train = UTKDataset(imgs_train, labels_train)
    ds_test = UTKDataset(imgs_test, labels_test)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=True)
    num_batches = ceil(len(ds_train) / args.batch_size)

    # Train the model
    model = VGGRegressionModel()
    optim = torch.optim.Adam(model.parameters())

    # Train loop
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
    ebar = tqdm(range(args.epochs), position=0)
    dl_iter = iter(dl_train)
    criterium = torch.nn.MSELoss()
    losses = []
    for epoch in range(args.epochs):
        batch_loss = []
        bbar = tqdm(total=len(dl_train), position=1, leave=False)
        batch_iter = iter(dl_train)
        for batchx, batchy in batch_iter:
            optim.zero_grad()

            pred = model(batchx)

            loss = criterium(pred, batchy)
            loss.backward()
            batch_loss.append(loss.item())
            optim.step()
            lr_scheduler.step()

            bbar.set_description(f"Batch Loss: {loss.item():.2f}")
            bbar.update()

        losses.append(sum(batch_loss) / len(batch_loss))
        lls = losses[-100:]
        ebar.set_description(f"Epoch Loss: {sum(lls)/len(lls):.2f}")
        ebar.update()
