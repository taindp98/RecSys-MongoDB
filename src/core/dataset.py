import cv2
import os
import torch
import json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transforms import get_transform
import pandas as pd
import torch
from PIL import Image
from load_config import load_config
from sklearn.model_selection import train_test_split


class FashionDataset(torch.utils.data.Dataset):
    def __init__(self, config, split="train", transforms=None):
        self.root = config.ROOT
        self.transforms = transforms
        self.config = config
        assert split in ["train", "val"]
        df_anno = pd.read_csv(os.path.join(config.ROOT, "train_labels.csv"))
        self.category_subset = json.load(
            open(os.path.join(config.ROOT, "trainval.json"), "r")
        )
        self.used_set = df_anno[df_anno["cloth_type"].isin(self.category_subset[split])]
        self.split = split
        if split == "val":
            gallery, query = train_test_split(
                self.used_set, test_size=0.2, random_state=42
            )
            gallery["is_query"] = 0
            query["is_query"] = 1
            self.used_set = pd.concat([gallery, query], axis=0)

    def __len__(self):
        return len(self.used_set)

    def _load_img(self, img: str):
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        record = self.used_set.iloc[idx]
        img_name = os.path.join(self.root, record["image_name"])
        image = self._load_img(img_name)
        cls_name = self.category_subset[self.split].index(record["cloth_type"])

        target = torch.tensor(cls_name, dtype=torch.long)
        if self.transforms:
            image = Image.fromarray(image)
            image = self.transforms(image)

        if self.split == "train":
            return (
                image,
                target,
                torch.tensor(0, dtype=torch.long),
                record["image_name"],
            )
        else:
            return (
                image,
                target,
                torch.tensor(record["is_query"], dtype=torch.long),
                record["image_name"],
            )


if __name__ == "__main__":

    cfg = load_config("../configs/local.yaml")
    train_ds = FashionDataset(
        split="train", config=cfg, transforms=get_transform(config=cfg, is_train=True)
    )

    train_dl = DataLoader(
        train_ds,
        sampler=RandomSampler(train_ds),
        batch_size=cfg.DATA.BATCH_SIZE,
        num_workers=cfg.DATA.NUM_WORKERS,
    )

    for batch in train_dl:
        images, targets = batch  ## [4,(3),3,224,224]
        print(images, targets)
        assert False
