import torch
import torch.nn as nn
from torchsummary import summary
from core.utils.mapper import configmapper
from core.modules.transforms import *
from core.modules.activations import *

def get_backbone(modal_encoder):
    if(modal_encoder.type=='resnet152'):
        model = torch.hub.load('pytorch/vision:v0.6.0','resnet152',pretrained=modal_encoder.params.pretrained)
    in_features = model.fc.in_features
    if(modal_encoder.params.remove_classifier):
      model = nn.Sequential(*list(model.children())[:-1])
    return model,in_features

def get_classifier(classifier):
    layers =[]
    if(classifier.custom_layers is None):
        if(classifier.type =='mlp'):
            for layer in range(classifier.params.num_layers):
                if(layer==0):
                    layers.append(nn.Linear(classifier.params.in_dim,classifier.params.hidden_dims[0]))
                    layers.append(configmapper.get_object('activations',classifier.params.activation.default.name)(**classifier.params.activation.default.params.as_dict()))
                    #print(layers)
                elif(layer==classifier.params.num_layers-1):
                    layers.append(nn.Linear(classifier.params.hidden_dims[-1],classifier.params.out_dim))
                    if(classifier.params.activation.output.name):
                        layers.append(configmapper.get_object('activations',classifier.params.activation.output.name)(**classifier.params.activation.output.params.as_dict()))
                    #print(layers)
                else:
                    layers.append(nn.Linear(classifier.params.hidden_dims[layer],classifier.params.hidden_dims[layer+1]))
                    layers.append(configmapper.get_object('activations',classifier.params.activation.default.name)(**classifier.params.activation.default.params.as_dict()))
    #print(layers)
    return nn.Sequential(*layers)

def map_dict_to_obj(dic):
    result_dic = {}
    if(dic is not None):
        for k,v in dic.items():
            if(isinstance(v,dict)):
                result_dic[k]=map_dict_to_obj(v)
            else:
                try:
                    obj = configmapper.get_object('params',v)
                    result_dic[k]=obj
                except:
                    result_dic[k]=v
    return result_dic

def get_image_processor(processor):
    transformations = []
    if(processor.type=='torchvision'):
        for param in processor.params:
            transformations.append(configmapper.get_object('transforms',param['type'])(**map_dict_to_obj(param['params'])))
    return transforms.Compose(transformations)
