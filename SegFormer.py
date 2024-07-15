from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch.nn as nn

class SegFormer(nn.Module):
    def __init__(self, num_labels=1):
        super(SegFormer, self).__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", num_labels=num_labels)
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits

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
