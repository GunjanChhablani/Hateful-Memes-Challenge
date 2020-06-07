import torch.nn as nn
from core.utils.mapper import configmapper
configmapper.map('activations','relu')(nn.ReLU)
configmapper.map('activations','logsoftmax')(nn.LogSoftmax)
configmapper.map('activations','softmax')(nn.Softmax)
