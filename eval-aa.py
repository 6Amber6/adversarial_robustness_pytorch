"""
Evaluation with AutoAttack.
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from autoattack import AutoAttack
    
from core.data import get_data_info
from core.data import load_data
from core.models import create_model

from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed



# Setup

parse = parser_eval()
args = parse.parse_args()
_cli_attack = args.attack
_cli_attack_eps = args.attack_eps

LOG_DIR = os.path.join(args.log_dir.rstrip('/'), args.desc)
with open(os.path.join(LOG_DIR, 'args.txt'), 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old)
# CLI overrides for eval (paper: Linf 8/255, version=standard)
if _cli_attack is not None:
    args.attack = _cli_attack
elif getattr(args, 'attack', None) is None:
    args.attack = 'linf-pgd'
if _cli_attack_eps is not None:
    args.attack_eps = _cli_attack_eps
elif getattr(args, 'attack_eps', None) is None:
    args.attack_eps = 8/255

# DATA_DIR: avoid double appending (args.data_dir may already be /path/data/cifar10s)
_data_dir = args.data_dir.rstrip('/')
if _data_dir.endswith(args.data):
    DATA_DIR = _data_dir
else:
    DATA_DIR = os.path.join(_data_dir, args.data)
LOG_DIR = os.path.join(args.log_dir.rstrip('/'), args.desc)
WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
log_path = os.path.join(LOG_DIR, 'log-aa.log')
logger = Logger(log_path)

info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.log('Using device: {}'.format(device))



# Load data

seed(args.seed)
_, _, train_dataloader, test_dataloader = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False, 
                                                    shuffle_train=False)

if args.train:
    logger.log('Evaluating on training set.')
    l = [x for (x, y) in train_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in train_dataloader]
    y_test = torch.cat(l, 0)
else:
    l = [x for (x, y) in test_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_dataloader]
    y_test = torch.cat(l, 0)



# Model

model = create_model(args.model, args.normalize, info, device)
checkpoint = torch.load(WEIGHTS)
if 'tau' in args and args.tau:
    print ('Using WA model.')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
del checkpoint



# AA Evaluation

seed(args.seed)
norm = 'Linf' if args.attack in ['fgsm', 'linf-pgd', 'linf-df'] else 'L2'
adversary = AutoAttack(model, norm=norm, eps=args.attack_eps, log_path=log_path, version=args.version, seed=args.seed)

if args.version == 'custom':
    adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
    adversary.apgd.n_restarts = 1
    adversary.apgd_targeted.n_restarts = 1

x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=BATCH_SIZE_VALIDATION)

print ('Script Completed.')