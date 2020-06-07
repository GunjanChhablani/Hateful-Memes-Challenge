import torchvision.transforms as transforms
from PIL import Image

from core.utils.mapper import configmapper
configmapper.map('transforms','Resize')(transforms.Resize)
configmapper.map('transforms','Normalize')(transforms.Normalize)
configmapper.map('transforms','ToTensor')(transforms.ToTensor)
configmapper.map('params','BICUBIC')(Image.BICUBIC)
