"""
AutoAttack evaluation for Parallel Fusion models.

Reads model config from args.txt in LOG_DIR, reconstructs
ParallelFusionWRNSwish, loads weights-best.pt, runs AutoAttack.
"""

import json
import os

import torch
import torch.nn as nn

from autoattack import AutoAttack

from core.data import get_data_info, load_data
from core.utils import Logger, parser_eval, seed
from core.utils.utils import str2float

from core.models.parallel_wrn_swish import create_parallel_fusion, get_class_splits


# Setup
parse = parser_eval()
args = parse.parse_args()
_cli_attack = args.attack
_cli_attack_eps = args.attack_eps

LOG_DIR = os.path.join(args.log_dir.rstrip('/'), args.desc)
with open(os.path.join(LOG_DIR, 'args.txt'), 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old)

# CLI overrides
if _cli_attack is not None:
    args.attack = _cli_attack
elif getattr(args, 'attack', None) is None:
    args.attack = 'linf-pgd'
if _cli_attack_eps is not None:
    args.attack_eps = _cli_attack_eps
elif getattr(args, 'attack_eps', None) is None:
    args.attack_eps = 8/255

# DATA_DIR
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
BATCH_SIZE_VALIDATION = args.batch_size_validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.log(f'Using device: {device}')


# Load data
seed(args.seed)
_, _, train_dataloader, test_dataloader = load_data(
    DATA_DIR, args.batch_size, BATCH_SIZE_VALIDATION, use_augmentation=False, shuffle_train=False
)

if args.train:
    logger.log('Evaluating on training set.')
    x_test = torch.cat([x for x, y in train_dataloader], 0)
    y_test = torch.cat([y for x, y in train_dataloader], 0)
else:
    x_test = torch.cat([x for x, y in test_dataloader], 0)
    y_test = torch.cat([y for x, y in test_dataloader], 0)


# Build model
logger.log(f'Model type: {args.model_type}')
logger.log(f'WRN-{args.depth}-{args.width}-{args.act_fn}, groups={args.num_groups}')

class_splits = None
if hasattr(args, 'class_splits') and args.class_splits:
    # Convert string keys back if needed
    class_splits = {k: v for k, v in args.class_splits.items()}

fusion, class_splits = create_parallel_fusion(
    depth=args.depth, width=args.width, act_fn=args.act_fn,
    dataset=args.data, num_classes=info['num_classes'],
    class_splits=class_splits,
)

# Wrap in DataParallel (must match training)
model = nn.DataParallel(fusion)
model = model.to(device)

# Load checkpoint
logger.log(f'Loading weights from {WEIGHTS}')
checkpoint = torch.load(WEIGHTS, map_location=device)
if 'tau' in args.__dict__ and args.tau:
    logger.log('Using WA model.')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
del checkpoint

logger.log(f'Class splits: {list(class_splits.keys())}')
total_params = sum(p.numel() for p in model.parameters())
logger.log(f'Total parameters: {total_params/1e6:.1f}M')


# AutoAttack evaluation
seed(args.seed)
norm = 'Linf' if args.attack in ['fgsm', 'linf-pgd', 'linf-df'] else 'L2'
logger.log(f'AutoAttack: norm={norm}, eps={args.attack_eps:.6f}, version={args.version}')

adversary = AutoAttack(model, norm=norm, eps=args.attack_eps,
                       log_path=log_path, version=args.version, seed=args.seed)

if args.version == 'custom':
    adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
    adversary.apgd.n_restarts = 1
    adversary.apgd_targeted.n_restarts = 1

x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=BATCH_SIZE_VALIDATION)

print('Script Completed.')
