import torch
import torchvision.models as models
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model.save_pretrained("./bert-base-uncased")
tokenizer.save_pretrained("./bert-base-uncased")


model = models.resnet50(pretrained=True)

torch.save(model.state_dict(), "./resnet50.pth")
