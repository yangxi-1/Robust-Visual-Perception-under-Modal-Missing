import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)


import json
import torch
from config import get_config
from dataset.builder import build_dataloader
from models.encoders import ImageEncoder, TextEncoder
from models.multimodal_model import MultiModalModel
from engine.train import train_one_epoch
from engine.evaluate import evaluate
from engine.loss import build_loss
import os
from utils.plot import plot_curves, plot_missing_curve
from utils.exp_res import create_exp_dir

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
cfg = get_config()


device = torch.device(cfg["device"])

train_loader, test_loader, num_cls, task_type = build_dataloader(cfg)



# drop_probs = [0.0, 0.2, 0.4, 0.6, 0.8]
# # drop_probs=[0.6]

# results = {
#     "drop_prob": [],
#     "f1_micro": []
}

exp_dir = create_exp_dir(dataset=cfg["dataset"])
print(f"📁 Saving results to: {exp_dir}")

with open(os.path.join(exp_dir, "config.json"), "w") as f:
    json.dump(cfg, f, indent=4)

# for dp in drop_probs:
#     print(f"\n==== Drop Prob: {dp} ====")

#     cfg["drop_prob"] = dp

#     train_loader, test_loader, num_cls, task_type = build_dataloader(cfg)

#     img_encoder = ImageEncoder(out_dim=256)
#     txt_encoder = TextEncoder(out_dim=256)
#     model = MultiModalModel(
#         img_encoder,
#         txt_encoder,
#         dim=256,
#         num_classes=num_cls
#     ).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

#     # loss
#     if task_type == "multilabel":
#         criterion = torch.nn.BCEWithLogitsLoss()
#     else:
#         criterion = torch.nn.CrossEntropyLoss()

#     # ===== 训练 =====
#     for epoch in range(cfg["epochs"]):
#         train_one_epoch(model, train_loader, optimizer, criterion, device)

#     # ===== 测试 =====
#     metrics = evaluate(model, test_loader, device, task_type)

#     # ===== 记录 =====
#     results["drop_prob"].append(dp)

#     if task_type == "multilabel":
#         results["f1_micro"].append(metrics["f1_micro"])
#     else:
#         results["f1_micro"].append(metrics["f1"])  # 统一名字
# plot_missing_curve(results, save_dir=exp_dir, show=True)
# with open(os.path.join(exp_dir, "results.json"), "w") as f:
#     json.dump(results, f, indent=4)

img_encoder = ImageEncoder(out_dim=256)
txt_encoder = TextEncoder(out_dim=256)
model = MultiModalModel(
    img_encoder,
    txt_encoder,
    dim=256,
    num_classes=num_cls
).to(device)

if cfg["freeze_encoder"]:

    print("🔥 Freezing encoders...")

    for param in model.img_encoder.parameters():
        param.requires_grad = False

    for param in model.txt_encoder.parameters():
        param.requires_grad = False

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=cfg["lr"]
)
criterion = build_loss(cfg)

if cfg["dataset"] == "mm_imdb":
    history = {
        "loss": [],
        "subset_acc": [],
        "hamming_acc": [],
        "f1_micro": [],
        "f1_macro": []
    }
else:
    history = {
        "loss": [],
        "acc": [],
        "f1": []
    }

for epoch in range(cfg["epochs"]):
    print(f"\nEpoch {epoch+1}/{cfg['epochs']}")

    loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    metrics = evaluate(
        model,
        test_loader,
        device,
        task_type=cfg["task"]
    )

    # ===== 记录 =====
    history["loss"].append(loss)

    if task_type == "multilabel":
        history["subset_acc"].append(metrics["subset_acc"])
        history["hamming_acc"].append(metrics["hamming_acc"])
        history["f1_micro"].append(metrics["f1_micro"])
        history["f1_macro"].append(metrics["f1_macro"])
    else:
        history["acc"].append(metrics["acc"])
        history["f1"].append(metrics["f1"])


    print(cfg["dataset"]+f" Training Loss: {loss:.4f}")
    if task_type == "multilabel":
        print(
            f"Subset Acc: {metrics['subset_acc']:.4f} | "
            f"Hamming Acc: {metrics['hamming_acc']:.4f} | "
            f"F1-micro: {metrics['f1_micro']:.4f} | "
            f"F1-macro: {metrics['f1_macro']:.4f}"
        )
    else:
        print(
            f"Acc: {metrics['acc']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )

plot_curves(history, save_dir=exp_dir, show=True)
with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
    json.dump(history, f, indent=4)
