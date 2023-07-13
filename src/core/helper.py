import cv2
import numpy as np
import torch
from .model_pool import ModelEmbedding
import os
import pymongo
import torch.nn.functional as F

def resize_aspect_ratio(img, size, interp=cv2.INTER_LINEAR):
    """
    resize min edge to target size, keeping aspect ratio
    """
    if len(img.shape) == 2:
        h, w = img.shape
    elif len(img.shape) == 3:
        h, w, _ = img.shape
    else:
        return None
    if h > w:
        new_w = size
        new_h = h * new_w // w
    else:
        new_h = size
        new_w = w * new_h // h
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def create_tensor_from_img(img):
    img = np.transpose(img, (0, 3, 1, 2))
    img = img / 255
    imagenet_mean = np.asarray([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
    imagenet_std = np.asarray([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
    img = (img - imagenet_mean) / imagenet_std
    img = torch.FloatTensor(img)
    return img


def min_edge_crop(img, position="center"):
    """
    crop image base on min size
    :param img: image to be cropped
    :param position: where to crop the image
    :return: cropped image
    """
    assert position in [
        "center",
        "left",
        "right",
    ], "position must either be: left, center or right"

    h, w = img.shape[:2]

    if h == w:
        return img

    min_edge = min(h, w)
    if h > min_edge:
        if position == "left":
            img = img[:min_edge]
        elif position == "center":
            d = (h - min_edge) // 2
            img = img[d:-d] if d != 0 else img

            if h % 2 != 0:
                img = img[1:]
        else:
            img = img[-min_edge:]

    if w > min_edge:
        if position == "left":
            img = img[:, :min_edge]
        elif position == "center":
            d = (w - min_edge) // 2
            img = img[:, d:-d] if d != 0 else img

            if w % 2 != 0:
                img = img[:, 1:]
        else:
            img = img[:, -min_edge:]

    assert (
        img.shape[0] == img.shape[1]
    ), f"height and width must be the same, currently {img.shape[:2]}"
    return img


def read_image(img_path, target_size=128):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_aspect_ratio(img, target_size)
    img = min_edge_crop(img, "center")
    img = np.transpose(img, (2, 0, 1))
    img = img / 255
    imagenet_mean = np.asarray([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
    imagenet_std = np.asarray([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
    img = (img - imagenet_mean) / imagenet_std
    img = torch.FloatTensor(img)
    return img

def load_model(cfg):
    model = ModelEmbedding(config = cfg)
    ckpt = torch.load(
        # os.path.join(cfg.SAVE_DIR, "checkpoint.pth"),
        cfg.SAVE_DIR,
        map_location="cpu"
    )
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

def init_mongo(uri_string, db_name, collection_name):
    client = pymongo.MongoClient(uri_string)
    db = client[db_name]
    collection = db['collection_name']
    return collection

def search(uri_string, img_path):
    collection = init_mongo(
        uri_string=uri_string,
        db_name="simsearch",
        collection_name="feature512"
    )
    img = read_image(img_path, target_size=128)
    model = load_model()
    with torch.no_grad():
        vector_query = F.normalize(model(img),1)[0].detach(
            ).cpu().numpy().tolist()
        pipeline = [
            {
                "$search": {
                    "index": "feature512",
                    "knnBeta": {
                        "vector": vector_query,
                        "path": "embedding",
                        "k": 10
                    }
                }
            },
            {
                "$project": {
                    "embedding": 0,
                    "_id": 0,
                    'score': {
                        '$meta': 'searchScore'
                    }
                }
            },
        ]
        res = collection.aggregate(pipeline)
        list_sim = []
        for r in res:
            list_sim.append((r['image_name'], r['score']))
    return list_sim
