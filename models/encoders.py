import torch.nn as nn
from torchvision import models
from transformers import BertModel
import torch


class ImageEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        backbone = models.resnet50(pretrained=False)
        backbone.load_state_dict(torch.load("model//resnet50.pth"))

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(2048, out_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TextEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained("model//bert-base-uncased")
        self.fc = nn.Linear(768, out_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_feature = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_feature)
