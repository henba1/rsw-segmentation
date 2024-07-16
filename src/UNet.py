import segmentation_models_pytorch as smp
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, encoder_name="resnet34", use_pretrained=True):
        super(UNet, self).__init__()
        encoder_weights = "imagenet" if use_pretrained else None
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
        )
    
    def forward(self, x):
        return self.model(x)
    
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
