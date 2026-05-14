import torch.nn as nn


def build_loss(cfg):
    if cfg["task"] == "multilabel":
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()
