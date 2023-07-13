from dataset import FashionDataset
from load_config import load_config
from transforms import get_transform
import torch.nn.functional as F

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse
from engine import insert_feature_to_db
from model_pool import ModelEmbedding
import os
import pymongo
from dotenv import load_dotenv

load_dotenv()


def main(cfg):
    client = pymongo.MongoClient(os.getenv("MONGOLAB_URI"))
    db = client.simsearch
    vector_collection = db["feature512"]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Config: {cfg}")
    model = ModelEmbedding(config=cfg)
    ckpt = torch.load(os.path.join(cfg.SAVE_DIR, "checkpoint.pth"), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    dataset = {
        "train": FashionDataset(
            config=cfg,
            split="train",
            transforms=get_transform(config=cfg, is_train=False),
        ),
        "valid": FashionDataset(
            config=cfg,
            split="val",
            transforms=get_transform(config=cfg, is_train=False),
        ),
    }

    train_dl = DataLoader(
        dataset["train"],
        sampler=RandomSampler(dataset["train"]),  ##
        batch_size=cfg.DATA.BATCH_SIZE,
        num_workers=cfg.DATA.NUM_WORKERS,
    )

    valid_dl = DataLoader(
        dataset["valid"],
        sampler=SequentialSampler(dataset["valid"]),
        batch_size=cfg.DATA.BATCH_SIZE,
        num_workers=cfg.DATA.NUM_WORKERS,
    )

    insert_feature_to_db(
        collection=vector_collection, model=model, dataloader=train_dl, device=device
    )

    insert_feature_to_db(
        collection=vector_collection, model=model, dataloader=valid_dl, device=device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-cfg", default="../configs/local.yaml")
    args = parser.parse_args()
    cfg = load_config(args.trial_cfg)
    main(cfg)
