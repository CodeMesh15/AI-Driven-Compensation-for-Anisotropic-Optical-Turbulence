# Defines the CompensationNet CNN architecture.

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class CompensationNet(nn.Module):
    def __init__(self, num_output_params):
        """
        Initializes the Compensation Neural Network.
        
        Args:
            num_output_params (int): The number of Zernike coefficients
                                     the network should predict.
        """
        super(CompensationNet, self).__init__()
        
        # Load a pre-trained ResNet-34 model
        self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        # The input images are grayscale (1-channel), but ResNet expects
        # 3-channel (RGB) images. We will modify the first convolution
        # layer to accept 1-channel input.
        # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # New:
        # We need to adapt the first layer to take 1-channel input instead of 3
        # We'll copy the weights from the 'R' channel and average them.
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, 64, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=original_conv1.bias
        )
        # Average the weights from the 3 channels to create 1 channel
        self.resnet.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

        
        # Replace the final fully-connected layer (the "classifier")
        # with a new regression head that outputs our 'num_output_params'.
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_output_params)

    def forward(self, x):
        """Defines the forward pass."""
        # The input 'x' is expected to be [batch_size, 1, H, W]
        return self.resnet(x)
