import torch
import torch.nn as nn
import math

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()

        self.oup = oup
        
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class AgileEEG(nn.Module):
    def __init__(self, num_classes=6, chans=11, samples=2048, 
                 F1=8, D=2, kernel_temporal=64):
        
        super(AgileEEG, self).__init__()

        F2 = F1 * D

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_temporal), stride=1, 
                      padding=(0, kernel_temporal // 2), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F2, (chans, 1), stride=1, 
                      padding=0, groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(inplace=True)
        )
        
        self.main_path = nn.Sequential(
            nn.BatchNorm2d(F2),
            nn.ReLU(inplace=True),
            nn.Conv2d(F2, F2, (1, 3), padding=(0, 1), groups=F2, bias=False),
            nn.BatchNorm2d(F2),
            GhostModule(F2, F2, kernel_size=1, ratio=2, dw_size=3),
            nn.BatchNorm2d(F2)
        )
        
        self.skip_path = nn.Sequential(
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2)
        )
        
        self.merge_relu = nn.ReLU(inplace=True)

        self.pooling = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        )

        final_samples = samples // 4 // 8
        self.final_features = F2 * final_samples
        
        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        
        x = self.spatial_conv(x)
        
        x_main = self.main_path(x)
        x_skip = self.skip_path(x)
        x = self.merge_relu(x_main + x_skip)
        
        x = self.pooling(x)
        
        x = self.classification(x)
        
        return x