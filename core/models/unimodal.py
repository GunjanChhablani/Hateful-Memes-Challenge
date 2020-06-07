import torch
import torch.nn as nn
from core.utils.mapper import configmapper
from core.utils.configuration import Config
from core.modules.builder import get_backbone,get_classifier,get_image_processor

@configmapper.map("models","unimodal")
class Unimodal(nn.Module):
  def __init__(self,config):
    super(Unimodal,self).__init__()
    self._config = config
    self.mode = config.mode
    if(self.mode == 'image'):
        self.modal_encoder,in_features = get_backbone(config.modal_encoder)
        self.classifier = get_classifier(config.classifier)
        self.flatten = nn.Flatten()

  def forward(self,x):
    if(self.mode=='image'):
        x = self.flatten(self.modal_encoder(x))
        x = self.classifier(x)
    return x
