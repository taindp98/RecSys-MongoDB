import os
import pymongo
import gradio as gr
from pathlib import Path
from mega import Mega

from dotenv import load_dotenv

import torch.nn.functional as F
import torch
from huggingface_hub import hf_hub_download

# BASE_DIR = Path(__file__).resolve().parent.parent
# import sys
# sys.path.append(str(BASE_DIR))
from src.core.helper import load_model, read_image
from src.core.load_config import load_config

load_dotenv()
mega = Mega()
mega = mega.login(os.getenv("mega_email"), os.getenv("mega_password"))

def main(img_path):
    if img_path:
        
        client = pymongo.MongoClient(os.getenv("MONGOLAB_URI"))
        db = client.simsearch
        collection = db['feature512']
        img = read_image(img_path, target_size=cfg.DATA.IMG_SIZE)
        
        with torch.no_grad():
            feat =  F.normalize(model(img),1)
            vector_query = feat[0].detach().cpu().numpy().tolist()
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
            list_retrieval = []
            for r in res:
                list_retrieval.append((r['image_name'], r['score']))
        ## download from mega
        list_fnames = []
        for item in list_retrieval:
            cat, fname = item[0].replace('\\','/').split('/')
            img_url = hf_hub_download(
                token=os.getenv("HUGGINGFACE"),
                repo_id="taindp98/fashion-recsys",
                filename=fname,
                subfolder=cat,
                repo_type="dataset",
            )
            print(f"img_url: {img_url}")
            list_fnames.append(img_url)
        return list_fnames

if __name__ == '__main__':
    
    model_local_path = hf_hub_download(
        token=os.getenv("HUGGINGFACE"),
        repo_id="taindp98/siamese-model",
        filename="checkpoint.pth",
        repo_type="dataset"
        )
    cfg = load_config("configs/deploy.yaml")
    cfg.defrost()
    cfg.SAVE_DIR = model_local_path
    cfg.freeze()
    model = load_model(cfg)
    load_dotenv()
    inputs_image = [
        gr.Image(
            type='filepath',
            label='Input Image'
        )
    ]
    outputs_image = [
        gr.Gallery(
            label='Search Results',
        ).style(
            columns=[5],
            object_fit='contain',
            height='auto'
        )
    ]
    demo = gr.Interface(
        fn = main,
        inputs=inputs_image,
        outputs=outputs_image,
        title='Demo Fashion Image Search',
        examples=[
            "./resources/blazer.jpg",
            "./resources/dress.jpg",
            "./resources/trouser.jpg"
        ]
    )

    demo.launch()