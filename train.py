## Imports
import argparse
import torch
import torch.nn as nn
from core.utils.misc import seed_all
from core.utils.reader import JsonlReader
from core.utils.configuration import Config
from core.utils.dataset import MemesDataset
from core.trainers.trainer import Trainer
from core.utils.mapper import configmapper
from core.models.unimodal import Unimodal
from torchsummary import summary
## Config
parser = argparse.ArgumentParser(prog = 'train.py',description='Train a model.')
parser.add_argument('--default',type=str,action='store',help='The main configuration file to be used.',default='./configs/default.yaml')
parser.add_argument('--model',type=str,action='store',help='The configuration for model',default='./configs/models/unimodal/image.yaml')
parser.add_argument('--trainer',type=str,action='store',help='The configuration for model training/evaluation',default='./configs/trainer.yaml')
parser.add_argument('--data',type=str,action='store',help='The configuration for data',default='./configs/data.yaml')
parser.add_argument('--demo',action='store_true',help='Whether to run a demo on CPU',default=False)
#parser.add_argument('--verbose',action='store_true',help='Whether or not to show training progress',default=True)
### Update Tips : Can provide more options to the user.
### Can also provide multiple verbosity levels.

args = parser.parse_args()
# print(vars(args))
if(not args.demo):
    main_config = Config(path = args.default)
    model_config = Config(path = args.model)
    trainer_config = Config(path = args.trainer)
    data_config = Config(path=args.data)
else:
    main_config = Config(path = './configs/demo/default.yaml')
    model_config = Config(path = './configs/demo/models/unimodal/image.yaml')
    trainer_config = Config(path = './configs/demo/trainer.yaml')
    data_config = Config(path='./configs/demo/data.yaml')

#verbose = args.verbose

## Seed
seed_all(42)

## Dataset
train_data = MemesDataset(data_config,'train')
dev_data = MemesDataset(data_config,'dev')

## Model
model = configmapper.get_object('models',model_config.name)(model_config)
print(summary(model,(3,224,224)))
## Trainer
trainer = configmapper.get_object('trainers',trainer_config.name)(trainer_config)

## Train
trainer.train(model,train_data,verbose=True,eval_dataset=dev_data)
