from torch.optim import AdamW
from core.utils.mapper import configmapper

configmapper.map('optimizers','adam_w')(AdamW)
