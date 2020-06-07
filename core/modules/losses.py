from torch.nn import CrossEntropyLoss
from core.utils.mapper import configmapper

configmapper.map('losses','cross_entropy')(CrossEntropyLoss)
