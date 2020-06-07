from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from core.utils.mapper import configmapper
configmapper.map('schedulers','cosine_warm')(CosineAnnealingWarmRestarts)
