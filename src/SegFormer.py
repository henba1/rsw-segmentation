import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch.nn as nn
import torch

class SegFormer(nn.Module):
    def __init__(self, num_labels=1, pretrained_model_name="nvidia/segformer-b0-finetuned-ade-512-512", use_pretrained=True):
        super(SegFormer, self).__init__()
        if use_pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(pretrained_model_name)
        else:
            # Load the configuration from the pretrained model name
            config = SegformerForSemanticSegmentation.config_class.from_pretrained(pretrained_model_name)
            config.num_labels = num_labels  # Update the number of labels
            self.model = SegformerForSemanticSegmentation(config)
            self.model.apply(self._init_weights)  # Initialize weights if not using pretrained
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(pretrained_model_name)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Repeat the single channel to create pseudo-RGB image
        if (x.shape[1] == 1):
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
