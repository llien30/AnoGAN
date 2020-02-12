import torch
from torch.utils.data import DataLoader

import yaml
from addict import Dict
import os
import argparse

from libs.dataloader import Dataset
from libs.transform import ImageTransform
from libs.weights import weights_init
from trainer import train_model

import wandb

def get_parameters():
    '''
    make parser to get parameters
    '''

    parser = argparse.ArgumentParser(
        description='take parameters like num_epochs ...')

    parser.add_argument('config', type=str, help='path of a config file')

    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Add --no_wandb option'
    )

    return parser.parse_args()

args = get_parameters()

CONFIG = Dict(yaml.safe_load(open(args.config)))

if not args.no_wandb:
    wandb.init(
        config = CONFIG,
        name = CONFIG.name,
        project = 'mnist',  #have to change when you want to change project

    )