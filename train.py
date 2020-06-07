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

## Config
parser = argparse.ArgumentParser(prog = 'train.py',description='Train a model.')
parser.add_argument('--default',type=str,action='store',help='The main configuration file to be used.',default='./configs/default.yaml')
parser.add_argument('--model',type=str,action='store',help='The configuration for model',default='./configs/models/unimodal/image.yaml')
parser.add_argument('--trainer',type=str,action='store',help='The configuration for model training/evaluation',default='./configs/trainer.yaml')
parser.add_argument('--data',type=str,action='store',help='The configuration for data',default='./configs/data.yaml')
#parser.add_argument('--verbose',action='store_true',help='Whether or not to show training progress',default=True)
### Update Tips : Can provide more options to the user.
### Can also provide multiple verbosity levels.

args = parser.parse_args()
# print(vars(args))
print(type(args.default))
main_config = Config(path = args.default)
model_config = Config(path = args.model)
trainer_config = Config(path = args.trainer)
data_config = Config(path=args.data)
#verbose = args.verbose

## Seed
seed_all(42)

## Dataset
train_data = MemesDataset(data_config,'train')
dev_data = MemesDataset(data_config,'dev')

## Model
model = configmapper.get_object('models',model_config.name)(model_config)

## Trainer
trainer = configmapper.get_object('trainers',trainer_config.name)(trainer_config)

## Train
trainer.train(model,train_data,verbose=True)
