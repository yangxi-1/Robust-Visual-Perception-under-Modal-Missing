import torch
import random
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random
from datasets import concatenate_datasets


class HatefulMemesDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, transform, img_dir=None, drop_prob=0.3):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.img_dir = img_dir
        self.drop_prob = drop_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # =========================
        # 1️⃣ 图像（🔥关键：兼容不同格式）
        # =========================
        if "image" in item and item["image"] is not None:
            image = item["image"]

        elif "img" in item and self.img_dir is not None:
            img_path = os.path.join(self.img_dir, item["img"])
            image = Image.open(img_path).convert("RGB")

        else:
            raise ValueError("No valid image field found")

        # 🔥 强制转 RGB（关键修复）
        image = image.convert("RGB")

        image = self.transform(image).float()

        # =========================
        # 2️⃣ 文本
        # =========================
        text = item["text"]

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # =========================
        # 3️⃣ label（二分类）
        # =========================
        label = torch.tensor(item["label"]).long()

        # =========================
        # 4️⃣ 模态缺失（🔥与你模型一致）
        # =========================
        img_exists, txt_exists = 1, 1

        if random.random() < self.drop_prob:
            img_exists = 0
            image = torch.zeros_like(image)

        if random.random() < self.drop_prob:
            txt_exists = 0
            input_ids = torch.zeros_like(input_ids)
            attention_mask = torch.zeros_like(attention_mask)

        mask = torch.tensor([img_exists, txt_exists])

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mask": mask,
            "label": label
        }


def build_memes(cfg):
    dataset = load_dataset("hateful_memes")

    tokenizer = BertTokenizer.from_pretrained("model/bert-base-uncased")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train_dataset = HatefulMemesDataset(
        dataset["train"], tokenizer, transform, drop_prob=cfg["drop_prob"]
    )

    test_split = concatenate_datasets([
        dataset["dev_seen"],
        dataset["dev_unseen"]
    ])

    test_dataset = HatefulMemesDataset(
        test_split, tokenizer, transform, drop_prob=cfg["drop_prob"]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg["batch_size"], shuffle=False)

    meta = {
        "num_classes": 2,
        "task": "classification"
    }

    return train_loader, test_loader, meta["num_classes"], meta["task"]
