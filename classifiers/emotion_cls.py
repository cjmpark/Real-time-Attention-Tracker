import torch.nn as nn
import torchvision.models as models

class ExpressionEvaluator(nn.Module):
    def __init__(self, num_emotions = 7):
        super().__init__()
        self.base = models.mobilenet_v2(pretrained=True) #pretrained on FER2013
        self.base.classifier[1] = nn.Linear(self.base.last_channel,num_emotions)
    
    def forward(self, x):
        return self.base(x)


