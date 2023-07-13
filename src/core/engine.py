import torch
import numpy as np
from utils import AverageMeter
from tqdm import tqdm
import torch.nn.functional as F


def train_one_epoch(
    cfg,
    model,
    dataloader,
    optimizer,
    lr_scheduler,
    epoch,
    loss_fnc,
    device=torch.device("cpu"),
):
    model.to(device)
    model.train()
    summary_loss = AverageMeter()
    tk0 = tqdm(dataloader, total=len(dataloader))
    for step, (images, targets, _, _) in enumerate(tk0):
        optimizer.zero_grad()
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        losses, _, _ = loss_fnc(logits, targets)
        losses.backward()
        optimizer.step()
        lr_scheduler.step_update(epoch * len(dataloader) + step)
        summary_loss.update(losses.item(), images.size(0))
        tk0.set_postfix(loss=summary_loss.avg, lr=optimizer.param_groups[0]["lr"])


def evaluate_one(cfg, model, valid_dataloader, device=torch.device("cpu")):
    """ """
    model.to(device)
    model.eval()

    gallery = []
    query = []

    tp_top1 = 0
    tp_topk = 0

    acc = {}
    k = 3
    with torch.no_grad():
        for images, targets, is_queries, img_info in tqdm(
            valid_dataloader, total=len(valid_dataloader)
        ):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            targets = targets.cpu().numpy()
            outputs = F.normalize(model(images), 1).detach().cpu().numpy()

            for target, output, is_query in zip(targets, outputs, is_queries):
                merge_array = np.concatenate([np.array([target]), output])
                if int(is_query) == 0:
                    gallery.append(merge_array)
                else:
                    query.append(merge_array)
        gallery = torch.tensor(np.array(gallery))
        query = torch.tensor(np.array(query))
        for q in query:
            sim = F.cosine_similarity(q[1:], gallery[:, 1:], 1)
            sorted_sim_idx = np.argpartition(sim.detach().cpu().numpy(), -k)[
                -k:
            ].tolist()  ## [273 477 784]
            sorted_sim = np.partition(sim.detach().cpu().numpy(), -k)[
                -k:
            ].tolist()  ## [0.98478818 0.9898822  0.993272  ]
            ## top1 acc
            pred_cat_topk = gallery[sorted_sim_idx][:, 0].numpy().tolist()
            targ_cat = q[0].item()
            if targ_cat == pred_cat_topk[-1]:
                tp_top1 += 1
            if targ_cat in pred_cat_topk:
                tp_topk += 1
        acc["top1"] = tp_top1 / len(query)
        acc[f"top{k}"] = tp_topk / len(query)

    return acc


def insert_feature_to_db(collection, model, dataloader, device=torch.device("cpu")):
    """ """
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, targets, is_queries, info_imgs in tqdm(
            dataloader, total=len(dataloader)
        ):
            images = images.to(device, non_blocking=True)
            outputs = F.normalize(model(images)).detach().cpu().numpy()

            for inf, output in zip(info_imgs, outputs):
                collection.insert_one({"image_name": inf, "embedding": output.tolist()})
        print("Insert done...")
