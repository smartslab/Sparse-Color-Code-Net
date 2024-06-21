import json
import yaml
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DownBlock, self).__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=2, padding=int((kernel_size-1)/2), groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.pointwise(x)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UNet(nn.Module):
    def __init__(self, num_depth, num_basefilter, input_channels=3, output_channels=1, kernel_size_down=3, kernel_size_up=3):
        super(UNet, self).__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels, num_basefilter, kernel_size=kernel_size_down, stride=1, padding=int((kernel_size_down-1)/2)),
            nn.BatchNorm2d(num_basefilter),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.down_blocks = nn.ModuleList()
        for i in range(num_depth):
            self.down_blocks.append(DownBlock(num_basefilter * (2**i), num_basefilter * (2**(i+1)), kernel_size_down))

        self.up_blocks = nn.ModuleList()
        for i in range(num_depth-1, 0, -1):
            self.up_blocks.append(UpBlock(num_basefilter * (2**(i+1)) + num_basefilter * (2**i), num_basefilter * (2**i), kernel_size_up))

        self.final_conv = nn.Sequential(
            nn.Conv2d(3*num_basefilter + input_channels, output_channels, kernel_size=1, stride=1)
        )
        _ = 0

    def forward(self, x):
        skip_connections = []
        orig_x = x
        x = self.init_conv(x)
        skip_connections.append(x)

        for block in self.down_blocks:
            x = block(x)
            skip_connections.append(x)

        # Reverse the skip connections for easy indexing during up-sampling
        skip_connections = skip_connections[::-1]

        # Up-sampling
        for i, block in enumerate(self.up_blocks):
            x = block(x, skip_connections[i+1])

        # Final convolution with skip from the initial conv and original input
        x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x)
        x = torch.cat((x, skip_connections[-1], orig_x), dim=1)
        fullsize_feature = x
        x = self.final_conv(x)

        return x, fullsize_feature

def initialize_weights(model):
    torch.manual_seed(1)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):  # Check if the module is a convolutional or linear layer
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')  # Apply He initialization
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)  # Optionally, set biases to zero
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

def DICELoss(inputs, targets, smooth=1.0):

    inputs = F.softmax(inputs, dim=1)

    inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
    targets = targets.view(targets.size(0), targets.size(1), -1)

    intersection = (inputs * targets).sum(-1)
    union = inputs.sum(-1) + targets.sum(-1)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    loss = 1 - dice.mean()

    return loss

def TverskyLoss(inputs, targets, alpha, beta):
    """
    Compute the Tversky loss with mean across the batch using .mean() method.

    Args:
    inputs (torch.Tensor): The raw outputs from the neural network, shape (batch_size, num_classes, H, W).
    targets (torch.Tensor): The binary maps for targets, shape (batch_size, num_classes, H, W).
    alpha (float): Weight for false positives.
    beta (float): Weight for false negatives.

    Returns:
    torch.Tensor: Computed Tversky loss with mean.
    """
    # Apply softmax to the inputs to get probabilities
    inputs = F.softmax(inputs, dim=1)

    # leave out the background class
    inputs = inputs[:,:-1,:,:]
    targets = targets[:,:-1,:,:]

    # True positives, false positives, and false negatives
    TP = (inputs * targets).sum(dim=[2, 3])
    FP = (inputs * (1 - targets)).sum(dim=[2, 3])
    FN = ((1 - inputs) * targets).sum(dim=[2, 3])

    # Tversky index for each sample in the batch
    Tversky = (TP + 1e-10) / (TP + alpha * FP + beta * FN + 1e-10)

    # Take mean over the batch
    return (1 - Tversky).mean()

def FocalLoss(inputs, targets, weight=None, gamma=0, eps=1e-7):
    """
    Compute the Focal Loss.
    Args:
    - inputs (tensor): Predictions from the model.
    - targets (tensor): Ground truth labels.
    - weight (tensor, optional): Weights for each class.
    - reduction is mean by defualt
    - gamma (float, optional): Focusing parameter.
    - eps (float, optional): Small value to ensure numerical stability.
    """
    ce_loss = F.cross_entropy(inputs, targets, weight=weight, reduction='none')
    p = torch.exp(-ce_loss)
    focal_loss = (1 - p) ** gamma * ce_loss


    return focal_loss.mean()


