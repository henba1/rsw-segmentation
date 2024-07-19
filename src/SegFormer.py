import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch.nn as nn
import torch

class SegFormer(nn.Module):
    def __init__(self, num_labels=1):
        super(SegFormer, self).__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512", 
            num_labels=num_labels, 
            ignore_mismatched_sizes=True
        )
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    def forward(self, x):
        # Repeat the single channel to create pseudo-RGB image
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        outputs = self.model(x)
        # Resize the output to match the input size
        logits = outputs.logits
        logits = F.interpolate(logits, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return logits

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def to(self, device):
        self.model.to(device)

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()
