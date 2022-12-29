import sys

sys.path.insert(0, '../src')

import torch
from mtools import vsummary
from torchvision.models import resnet50

if __name__ == '__main__':
    model = resnet50()
    input = torch.randn((1, 3, 224, 224))
    vsummary(model, input_data=(input,), depth=2)