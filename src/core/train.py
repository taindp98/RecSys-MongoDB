from dataset import FashionDataset
from load_config import load_config
from transforms import get_transform

import torch
from optimizer import build_optimizer
from scheduler import build_scheduler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse
from engine import train_one_epoch, evaluate_one
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from model_pool import ModelEmbedding

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from perte import OnlineTripletLoss
from perte import AllTripletSelector
import os


def main(cfg):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = ModelEmbedding(config=cfg)
    dataset = {
        "train": FashionDataset(
            config=cfg,
            split="train",
            transforms=get_transform(config=cfg, is_train=True),
        ),
        "valid": FashionDataset(
            config=cfg,
            split="val",
            transforms=get_transform(config=cfg, is_train=False),
        ),
    }
    train_dl = DataLoader(
        dataset["train"],
        sampler=RandomSampler(dataset["train"]),
        batch_size=cfg.DATA.BATCH_SIZE,
        num_workers=cfg.DATA.NUM_WORKERS,
    )

    valid_dl = DataLoader(
        dataset["valid"],
        sampler=SequentialSampler(dataset["valid"]),
        batch_size=cfg.DATA.BATCH_SIZE,
        num_workers=cfg.DATA.NUM_WORKERS,
    )

    optimizer = build_optimizer(model=model, opt_func=cfg.OPT_NAME, lr=cfg.BASE_LR)
    lr_scheduler = build_scheduler(
        config=cfg, optimizer=optimizer, n_iter_per_epoch=len(train_dl)
    )
    if cfg.TEST_ONLY:
        model = torch.load(
            os.path.join(cfg.SAVE_DIR, "checkpoint.pth"), map_location="cpu"
        )
        eval_metrics = evaluate_one(
            cfg=cfg, model=model, valid_dataloader=valid_dl, device=device
        )
        print(f"Test results: {eval_metrics}")
        return
    triplet_selector = AllTripletSelector()
    loss_fnc = OnlineTripletLoss(
        triplet_selector=triplet_selector, margin=0.5, reduction="mean", device=device
    )
    best_score = 0.0
    for epoch in range(cfg.NUM_EPOCHS):
        print(f"Training epoch: {epoch}")
        train_one_epoch(
            cfg=cfg,
            model=model,
            dataloader=train_dl,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            loss_fnc=loss_fnc,
            device=device,
        )
        if (epoch + 1) % cfg.FREQ_EVAL == 0:
            eval_metrics = evaluate_one(
                cfg=cfg, model=model, valid_dataloader=valid_dl, device=device
            )
            print(f"Valid results: {eval_metrics}")

            if best_score < eval_metrics["top1"]:
                os.makedirs(cfg.SAVE_DIR, exist_ok=True)
                best_score = eval_metrics["top1"]
                torch.save(
                    {
                        "model": model.state_dict(),
                    },
                    os.path.join(cfg.SAVE_DIR, "checkpoint.pth"),
                )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-cfg", default="../configs/local.yaml")
    args = parser.parse_args()
    cfg = load_config(args.trial_cfg)
    main(cfg)
