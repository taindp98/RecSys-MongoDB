import torch.nn as nn
import timm


def build_head(in_fts, out_fts, is_complex=False):
    if is_complex:
        print("Build complex MLP head")
        head = nn.Sequential(
            nn.Linear(in_fts, in_fts // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(in_fts // 4),
            nn.Linear(in_fts // 4, out_fts),
        )
    else:
        print("Build simple MLP head")
        head = nn.Linear(in_fts, out_fts, bias=True)
    return head


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


class ModelEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        if str(config.MODEL.BACKBONE).startswith("resnet"):
            model = timm.create_model(config.MODEL.BACKBONE, pretrained=True)
            in_fts = model.fc.in_features
            mlp = build_head(in_fts, config.MODEL.EMB_SIZE)
        elif str(config.MODEL.BACKBONE).startswith("densenet") or str(
            config.MODEL.BACKBONE
        ).startswith("efficientnet"):
            model = timm.create_model(config.MODEL.BACKBONE, pretrained=True)
            in_fts = model.classifier.in_features
            mlp = build_head(in_fts, config.MODEL.EMB_SIZE)

        backbone = nn.Sequential(*(list(model.children())[:-1]))
        self.model = nn.Sequential(backbone, mlp)

    def forward(self, x):
        return self.model(x)
