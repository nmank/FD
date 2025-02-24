import torch
from torch import nn

class AlexNetLastTwoLayers(nn.Module):
    def __init__(self, model):
        super(AlexNetLastTwoLayers, self).__init__()
        # Copy the feature extraction and pooling layers
        self.features = model.features
        self.avgpool = model.avgpool
        self.fc0 = model.classifier[:2]
        self.fc1 = model.classifier[2:4]
        self.fc2 = model.classifier[4:6]

    def forward(self, x):
        # Pass through feature extractor
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc0(x)
        x_fc1 = self.fc1(x)  # Output of the second-to-last layer
        x_fc2 = self.fc2(x_fc1)  # Output of the last layer before final output
        
        return x_fc1, x_fc2