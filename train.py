from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src.dataset import MedicalImageDataset, LoadImaged, ToTensord, ResizeImaged
from src.metric import DiceScore
# from src.model import YOUR_MODEL
# from src.loss import LOSS_FUNC
from src.trainer import Trainer
from src.utils import set_random_seed


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='cnn_unet')
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()
    config = OmegaConf.load(Path("configs", f"{args.config}.yaml"))
    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ToTensord(keys=["image", "label"]),
            ResizeImaged(keys=["image", "label"], **config.transform.resized_imaged),
        ]
    )

    valid_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ToTensord(keys=["image", "label"]),
            ResizeImaged(keys=["image"], **config.transform.resized_imaged),
        ]
    )

    train_dataset = MedicalImageDataset(**config.train_dataset, transform=train_transform)
    train_loader = DataLoader(train_dataset, **config.train_loader)
    valid_dataset = MedicalImageDataset(**config.valid_dataset, transform=valid_transform)
    valid_loader = DataLoader(valid_dataset, **config.valid_loader)

    #TODO: Prepare your model
    model = YOUR_MODEL

    #TODO: Prepare your loss func
    loss_func = LOSS_FUNC

    #TODO: Define your optimizer
    optimizer = YOUR_OPTIMIZER

    #TODO: Define your learning rate scheduler
    lr_scheduler = YOUR_LR_SCHEDULER

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="ntucsie_ml_hw2",)
    wandb.watch(model, log="all")

    trainer = Trainer(
        model=model.to(device),
        device=device,
        criterion=loss_func,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        eval_func=DiceScore(),
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_path=config.checkpoint.path,
        logger=wandb,
    )
    trainer.fit(**config.trainer)
