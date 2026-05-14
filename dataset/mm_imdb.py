import torch
import random
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset


class MMIMDbDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, transform, drop_prob):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.drop_prob = drop_prob

        # ===== label space =====
        labels_set = set()
        for item in hf_dataset:
            for l in item["labels"]:
                labels_set.add(l)

        self.label2id = {l: i for i, l in enumerate(sorted(labels_set))}
        self.num_classes = len(self.label2id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # ===== image =====
        image = self.transform(item["image"])

        # ===== text =====
        text = item["text"]

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # ===== multi-label =====
        label = torch.zeros(self.num_classes)
        for l in item["labels"]:
            label[self.label2id[l]] = 1

        # ===== mask =====
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


def build_mmimdb(cfg):
    dataset = load_dataset("MM-IMDb")

    full = dataset["train"]
    split = full.train_test_split(test_size=0.1, seed=42)

    tokenizer = BertTokenizer.from_pretrained("model/bert-base-uncased")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train_dataset = MMIMDbDataset(
        split["train"], tokenizer, transform, drop_prob=cfg["drop_prob"])
    test_dataset = MMIMDbDataset(
        split["test"], tokenizer, transform, drop_prob=cfg["drop_prob"])

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg["batch_size"], shuffle=False)

    meta = {
        "num_classes": train_dataset.num_classes,
        "task": "multilabel"
    }

    return train_loader, test_loader, meta["num_classes"], meta['task']
