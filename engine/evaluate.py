from tqdm import tqdm
from sklearn.metrics import f1_score
import torch

def evaluate(model, loader, device, task_type="multilabel"):
    model.eval()

    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in pbar:
            image = batch["image"].to(device)
            text_inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)

            output = model(image, text_inputs, mask)

            # =========================
            # 🎯 核心分叉点（就在这里）
            # =========================
            if task_type == "multilabel":
                pred = (torch.sigmoid(output) > 0.5).float()
            else:  # classification
                pred = output.argmax(dim=1)

            # =========================
            # 存储
            # =========================
            all_preds.append(pred.cpu())
            all_labels.append(label.cpu())

    # ===== 拼接 =====
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # =========================
    # 🎯 指标计算（也要分叉！）
    # =========================
    if task_type == "multilabel":
        subset_acc = (all_preds == all_labels).all(dim=1).float().mean().item()
        hamming_acc = (all_preds == all_labels).float().mean().item()

        f1_micro = f1_score(all_labels, all_preds, average='micro')
        f1_macro = f1_score(all_labels, all_preds, average='macro')

        return {
            "subset_acc": subset_acc,
            "hamming_acc": hamming_acc,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro
        }

    else:  # classification
        acc = (all_preds == all_labels).float().mean().item()
        f1 = f1_score(all_labels, all_preds, average='macro')

        return {
            "acc": acc,
            "f1": f1
        }