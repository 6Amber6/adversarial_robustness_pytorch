"""
Two-stage Parallel Fusion Adversarial Training with Weight Averaging (EMA).

Stage 1: Train sub-models on class subsets with CE loss.
Stage 2: Fuse with TRADES + EMA (identical to Gowal et al., 2020 / Rebuffi et al., 2021).

All Stage 2 hyperparameters match the baseline train-wa.py exactly.
Parallel-specific additions: backbone_lr_ratio, aux_ce_loss, warmup.
"""

import json
import time
import copy
import shutil
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms

from core.data import get_data_info, load_data, DATASETS, SEMISUP_DATASETS
from core.attacks import create_attack, CWLoss
from core.metrics import accuracy
from core.utils import format_time, Logger, seed
from core.utils import set_bn_momentum


def freeze_bn(model):
    """Freeze BN layers: set to eval mode and disable gradient."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def unfreeze_bn(model):
    """Unfreeze BN layers: set to train mode and enable gradient."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.train()
            for p in m.parameters():
                p.requires_grad = True
from core.utils import ctx_noparamgrad_and_eval
from core.utils import parser_train

from core.models.parallel_wrn_swish import (
    create_parallel_fusion, get_class_splits,
)

from gowal21uncovering.utils.watrain import ema_update, update_bn
from gowal21uncovering.utils.trades import trades_loss
from gowal21uncovering.utils.cutmix import cutmix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================================================
# Dataset wrapper for class subsets (Stage 1)
# =========================================================
class CIFARSubset(torch.utils.data.Dataset):
    """Filter dataset to specified classes, remap labels to 0..N-1."""
    def __init__(self, base_dataset, keep_classes):
        self.base = base_dataset
        self.map = {c: i for i, c in enumerate(keep_classes)}
        targets = torch.tensor(self.base.targets)
        self.idx = (
            (targets.unsqueeze(1) == torch.tensor(keep_classes))
            .any(dim=1).nonzero(as_tuple=False).view(-1)
        )

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x, y = self.base[self.idx[i]]
        return x, self.map[y]


# =========================================================
# Aux CE loss on sub-model logits
# =========================================================
def aux_ce_loss(sub_logits_list, y, class_splits, dev, weight=0.02):
    """CE on each sub-model using only samples from its class subset."""
    loss = torch.tensor(0.0, device=dev)
    for logits, (_, classes) in zip(sub_logits_list, class_splits.items()):
        ct = torch.tensor(classes, device=dev)
        mask = torch.isin(y, ct)
        if mask.any():
            local_y = torch.searchsorted(ct, y[mask])
            loss = loss + F.cross_entropy(logits[mask], local_y)
    return weight * loss


# =========================================================
# Backbone LR ratio schedule (parallel-specific)
# =========================================================
def backbone_lr_ratio(epoch, total_epochs):
    """Three-phase backbone LR ratio schedule.
    Phase 1 (0-20%):  0.15 — stabilize FC, backbones adapt slowly
    Phase 2 (20-65%): 0.50 — backbones adapt fully for adversarial features
    Phase 3 (65%+):   0.35 — stabilize for convergence (aligns with LR decay at 67%)
    """
    p1 = max(1, int(total_epochs * 0.2))
    p2 = max(p1 + 1, int(total_epochs * 0.65))
    if epoch <= p1:
        return 0.15
    if epoch <= p2:
        return 0.50
    return 0.35


# =========================================================
# Stage 1: train one sub-model for one epoch
# =========================================================
def train_ce_epoch(model, loader, optimizer, num_classes=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # CutMix with Beta(0.5, 0.5) — softer than Stage 2's Beta(1,1)
        if num_classes is not None:
            x, y = cutmix(x, y, num_classes, alpha=0.5, beta=0.5)
        optimizer.zero_grad()
        logits = model(x)
        if y.dim() == 2:
            # Soft labels from CutMix
            loss = -(y * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            y_hard = y.argmax(dim=1)
        else:
            loss = F.cross_entropy(logits, y)
            y_hard = y
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y_hard).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def eval_acc(model, loader, dev):
    """Evaluate clean accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            correct += (model(x).argmax(1) == y).sum().item()
            total += x.size(0)
    return correct / total


# =========================================================
# Args — extend baseline parser_train() with parallel-specific args
# =========================================================
parse = parser_train()

# Override defaults to make --data not required to be semisup
parse._option_string_actions['-d'].default = 'cifar10s'

# Parallel-specific
parse.add_argument('--tau', type=float, default=0.999, help='EMA decay.')
parse.add_argument('--depth', type=int, default=28, help='WRN depth.')
parse.add_argument('--width', type=int, default=10, help='WRN widen factor.')
parse.add_argument('--act-fn', type=str, default='swish', choices=['swish', 'relu'])
parse.add_argument('--num-groups', type=int, default=2,
                   help='Number of class groups (2 for CIFAR-10; 2 or 4 for CIFAR-100)')
parse.add_argument('--epochs-sub', type=int, default=100, help='Stage 1 CE epochs per sub-model.')
parse.add_argument('--epochs-warmup', type=int, default=10, help='Stage 2 CE warmup epochs.')
parse.add_argument('--aux-weight', type=float, default=0.02, help='Aux CE loss weight.')
parse.add_argument('--bn-freeze-epochs', type=int, default=-1,
                   help='Epochs to keep BN frozen after warmup. -1=permanent, 0=no freeze.')

args = parse.parse_args()

# Mark model type for eval scripts
args.model_type = 'parallel-fusion'
args.normalize = False  # Swish WRN has built-in normalization

# CutMix => step scheduler at 2/3 (same as baseline)
if args.cutmix:
    args.scheduler = 'step'
    args.scheduler_milestones = [max(1, int(0.667 * args.num_adv_epochs))]

# Paths
DATA_DIR = os.path.join(args.data_dir, args.data)
LOG_DIR = os.path.join(args.log_dir, args.desc)
WEIGHTS_BEST = os.path.join(LOG_DIR, 'weights-best.pt')
WEIGHTS_LAST = os.path.join(LOG_DIR, 'weights-last.pt')

if os.path.exists(LOG_DIR) and not os.path.exists(WEIGHTS_LAST):
    # Clean non-checkpoint files; keep sub-*.pt from Stage 1
    for f in os.listdir(LOG_DIR):
        fpath = os.path.join(LOG_DIR, f)
        if os.path.isfile(fpath) and not f.startswith('sub-'):
            os.remove(fpath)
os.makedirs(LOG_DIR, exist_ok=True)
logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)

info = get_data_info(DATA_DIR)
NUM_CLASSES = info['num_classes']
NUM_ADV_EPOCHS = args.num_adv_epochs

if args.debug:
    args.epochs_sub = 1
    args.epochs_warmup = 1
    NUM_ADV_EPOCHS = 1

seed(args.seed)
torch.backends.cudnn.benchmark = True

logger.log(f'Device: {device}')
logger.log(f'Dataset: {args.data}, classes: {NUM_CLASSES}')


# =========================================================
# Data loading
# =========================================================
class_splits = get_class_splits(args.data, args.num_groups)
logger.log(f'Class splits ({args.num_groups} groups):')
for name, cls in class_splits.items():
    logger.log(f'  {name}: {len(cls)} classes {cls[:5]}...' if len(cls) > 5 else f'  {name}: {len(cls)} classes {cls}')

# Full dataset for Stage 2 (same as baseline train-wa.py)
if args.data in SEMISUP_DATASETS:
    _, _, _, train_dataloader, test_dataloader, eval_dataloader = load_data(
        DATA_DIR, args.batch_size, args.batch_size_validation, use_augmentation=args.augment,
        shuffle_train=True, aux_data_filename=args.aux_data_filename,
        unsup_fraction=args.unsup_fraction, validation=True
    )
else:
    _, _, train_dataloader, test_dataloader = load_data(
        DATA_DIR, args.batch_size, args.batch_size_validation,
        use_augmentation=args.augment, shuffle_train=True
    )
    eval_dataloader = None

# Stage 1 sub-model datasets (standard CIFAR, simple augmentation)
sub_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
# Download standard CIFAR to data_dir (works even if using cifar10s/cifar100s for Stage 2)
if 'cifar100' in args.data:
    base_sub_dataset = torchvision.datasets.CIFAR100(
        root=os.path.join(args.data_dir, 'cifar100'), train=True, download=True, transform=sub_transform)
else:
    base_sub_dataset = torchvision.datasets.CIFAR10(
        root=os.path.join(args.data_dir, 'cifar10'), train=True, download=True, transform=sub_transform)

sub_loaders = {}
for name, classes in class_splits.items():
    subset = CIFARSubset(base_sub_dataset, classes)
    sub_loaders[name] = torch.utils.data.DataLoader(
        subset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )
    logger.log(f'  sub-loader {name}: {len(subset)} samples')

# Test subset loaders for Stage 1 evaluation
test_transform = transforms.Compose([transforms.ToTensor()])
if 'cifar100' in args.data:
    base_test_dataset = torchvision.datasets.CIFAR100(
        root=os.path.join(args.data_dir, 'cifar100'), train=False, download=True, transform=test_transform)
else:
    base_test_dataset = torchvision.datasets.CIFAR10(
        root=os.path.join(args.data_dir, 'cifar10'), train=False, download=True, transform=test_transform)
sub_test_loaders = {}
for name, classes in class_splits.items():
    test_subset = CIFARSubset(base_test_dataset, classes)
    sub_test_loaders[name] = torch.utils.data.DataLoader(
        test_subset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )


# =========================================================
# Model creation
# =========================================================
logger.log(f'\nModel: ParallelFusionWRN depth={args.depth} width={args.width} act={args.act_fn}')

fusion_raw, class_splits = create_parallel_fusion(
    depth=args.depth, width=args.width, act_fn=args.act_fn,
    dataset=args.data, num_classes=NUM_CLASSES, class_splits=class_splits,
)

# Save class_splits to args.txt (for eval script)
args.class_splits = {k: v for k, v in class_splits.items()}
with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)


# =========================================================
# Resume detection
# =========================================================
start_stage = 'sub'
start_epoch = 1
warmup_start = 1
old_score = [0.0, 0.0]
metrics = pd.DataFrame()
resume_ckpt = None

if os.path.exists(WEIGHTS_LAST):
    resume_ckpt = torch.load(WEIGHTS_LAST, map_location='cpu')
    if isinstance(resume_ckpt, dict) and 'stage' in resume_ckpt:
        start_stage = resume_ckpt['stage']
        old_score = resume_ckpt.get('old_score', [0.0, 0.0])
        if start_stage == 'warmup':
            warmup_start = resume_ckpt['epoch'] + 1
        elif start_stage == 'adv':
            start_epoch = resume_ckpt['epoch'] + 1
        logger.log(f'Resume detected: stage={start_stage}, epoch={resume_ckpt["epoch"]}')
        if os.path.exists(os.path.join(LOG_DIR, 'stats_adv.csv')):
            try:
                metrics = pd.read_csv(os.path.join(LOG_DIR, 'stats_adv.csv'))
            except Exception:
                pass


# =========================================================
# Stage 1: Train sub-models with CE
# =========================================================
if start_stage == 'sub':
    logger.log('\n======= Stage 1: Sub-model CE Pre-training =======')
    sub_models = list(fusion_raw.subs)

    for i, (name, classes) in enumerate(class_splits.items()):
        sub = sub_models[i]
        sub_path = os.path.join(LOG_DIR, f'sub-{name}.pt')

        if os.path.exists(sub_path):
            logger.log(f'  Loading pre-trained {name} from {sub_path}')
            sub.load_state_dict(torch.load(sub_path, map_location='cpu'))
            continue

        logger.log(f'  Training {name} ({len(classes)} classes, {args.epochs_sub} epochs)')
        for p in sub.parameters():
            p.requires_grad = True
        sub = sub.to(device)

        opt = optim.SGD(sub.parameters(), lr=0.1, momentum=0.9,
                        weight_decay=args.weight_decay, nesterov=True)
        sched = optim.lr_scheduler.MultiStepLR(
            opt, milestones=[max(1, int(args.epochs_sub * 0.6))], gamma=0.1
        )

        for ep in range(1, args.epochs_sub + 1):
            _, acc = train_ce_epoch(sub, sub_loaders[name], opt,
                                    num_classes=len(classes) if args.cutmix else None)
            sched.step()
            if ep % 10 == 0 or ep == args.epochs_sub:
                test_acc_sub = eval_acc(sub, sub_test_loaders[name], device)
                logger.log(f'    [{name}][{ep}/{args.epochs_sub}] train_acc={acc*100:.2f}% '
                           f'test_acc={test_acc_sub*100:.2f}% lr={opt.param_groups[0]["lr"]:.6f}')

        # Final test accuracy
        final_test = eval_acc(sub, sub_test_loaders[name], device)
        logger.log(f'  {name} done. Test Accuracy: {final_test*100:.2f}%')

        sub.cpu()
        torch.save(sub.state_dict(), sub_path)
        for p in sub.parameters():
            p.requires_grad = False

    start_stage = 'warmup'

elif resume_ckpt is not None:
    # Load sub-model weights from fusion checkpoint
    pass  # They're inside the fusion state_dict


# =========================================================
# Stage 2: Prepare fusion model with DataParallel + EMA
# =========================================================
logger.log('\n======= Stage 2: Fusion Training =======')

# Unfreeze all parameters
for p in fusion_raw.parameters():
    p.requires_grad = True

# Wrap in DataParallel (same as baseline)
model = nn.DataParallel(fusion_raw)
model = model.to(device)

# WA model (EMA copy, same as WATrainer)
wa_model = copy.deepcopy(model)

# Load fusion state if resuming
if resume_ckpt is not None and 'model_state_dict' in resume_ckpt:
    model.load_state_dict(resume_ckpt['model_state_dict'])
    logger.log('  Loaded model state from checkpoint')
if resume_ckpt is not None and 'wa_model_state_dict' in resume_ckpt:
    wa_model.load_state_dict(resume_ckpt['wa_model_state_dict'])
    logger.log('  Loaded WA model state from checkpoint')

# Optimizer: separate backbone (sub-models) vs FC (fusion layer), BN no weight decay
def group_weight_parallel(mdl):
    backbone_decay, backbone_no_decay = [], []
    fc_decay = []
    for n, p in mdl.named_parameters():
        is_fusion_fc = ('module.fc.' in n)
        if is_fusion_fc:
            fc_decay.append(p)
        elif 'batchnorm' in n:
            backbone_no_decay.append(p)
        else:
            backbone_decay.append(p)
    total = len(backbone_decay) + len(backbone_no_decay) + len(fc_decay)
    assert total == len(list(mdl.parameters())), f'Param count mismatch: {total} vs {len(list(mdl.parameters()))}'
    return [
        dict(params=backbone_decay, is_backbone=True),
        dict(params=backbone_no_decay, weight_decay=0.0, is_backbone=True),
        dict(params=fc_decay, is_backbone=False),
    ]

optimizer = optim.SGD(group_weight_parallel(model), lr=args.lr,
                      weight_decay=args.weight_decay, momentum=0.9, nesterov=args.nesterov)

# Eval attack on WA model (same as WATrainer)
eval_attack = create_attack(wa_model, CWLoss, args.attack, args.attack_eps,
                            4 * args.attack_iter, args.attack_step)

# EMA config (same as WATrainer)
num_samples = 50000  # CIFAR-10/100 train size
update_steps = int(np.floor(num_samples / args.batch_size) + 1)
warmup_steps_ema = 0.025 * NUM_ADV_EPOCHS * update_steps

# Scheduler
def init_scheduler(num_epochs):
    if args.scheduler == 'cosinew':
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start=0.025,
                                              total_steps=num_epochs)
    elif args.scheduler == 'step':
        ms = getattr(args, 'scheduler_milestones', None) or [100, 105]
        return optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=ms)
    elif args.scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif args.scheduler == 'cyclic':
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start=0.25,
                                              steps_per_epoch=update_steps, epochs=num_epochs)
    return None

# Resume optimizer
if resume_ckpt is not None and 'optimizer_state_dict' in resume_ckpt:
    try:
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        logger.log('  Resumed optimizer')
    except Exception as e:
        logger.log(f'  Could not resume optimizer: {e}')


# =========================================================
# Stage 2a: CE Warmup (parallel-specific, stabilizes FC layer)
# =========================================================
if start_stage == 'warmup' and args.epochs_warmup > 0:
    logger.log(f'\n--- CE Warmup ({warmup_start}-{args.epochs_warmup}) ---')

    for ep in range(warmup_start, args.epochs_warmup + 1):
        model.train()
        if ep == 1:
            set_bn_momentum(model, momentum=1.0)
        elif ep == 2:
            set_bn_momentum(model, momentum=0.01)

        train_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(train_dataloader, desc=f'Warmup {ep}', disable=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            if y.dim() == 2:
                loss = -(y * F.log_softmax(out, dim=1)).sum(dim=1).mean()
                y_acc = y.argmax(dim=1)
            else:
                loss = F.cross_entropy(out, y)
                y_acc = y
            loss.backward()
            if args.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            correct += (out.detach().argmax(1) == y_acc).sum().item()
            total += x.size(0)

        # Copy model to WA model during warmup
        for p_wa, p_m in zip(wa_model.parameters(), model.parameters()):
            p_wa.data.copy_(p_m.data)
        update_bn(wa_model, model)

        # Test accuracy (fusion model on full test set)
        test_acc_warmup = 0.0
        model.eval()
        with torch.no_grad():
            for x_t, y_t in test_dataloader:
                x_t, y_t = x_t.to(device), y_t.to(device)
                test_acc_warmup += accuracy(y_t, model(x_t))
        test_acc_warmup /= len(test_dataloader)

        logger.log(f'[Warmup][{ep}/{args.epochs_warmup}] loss={train_loss/total:.4f} '
                    f'train_acc={correct/total*100:.2f}% test_acc={test_acc_warmup*100:.2f}%')

        torch.save({
            'model_state_dict': model.state_dict(),
            'wa_model_state_dict': wa_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stage': 'warmup', 'epoch': ep, 'old_score': old_score,
        }, WEIGHTS_LAST)

start_stage = 'adv'
if start_epoch <= 1:
    start_epoch = 1

# Freeze BN after warmup (prevents adversarial examples from corrupting BN stats)
freeze_bn(model)
freeze_bn(wa_model)
logger.log('  BN frozen after warmup')

# Initialize scheduler for adversarial phase
scheduler = init_scheduler(NUM_ADV_EPOCHS)
if resume_ckpt is not None and 'scheduler_state_dict' in resume_ckpt and resume_ckpt.get('stage') == 'adv':
    try:
        scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        logger.log('  Resumed scheduler')
    except Exception as e:
        logger.log(f'  Could not resume scheduler: {e}')

del resume_ckpt  # free memory


# =========================================================
# Stage 2b: TRADES + EMA (matches baseline train-wa.py)
# =========================================================
# BN freeze schedule: -1=permanent, 0=no freeze, N=unfreeze at epoch N
if args.bn_freeze_epochs < 0:
    BN_UNFREEZE_EPOCH = NUM_ADV_EPOCHS + 1  # never unfreeze
elif args.bn_freeze_epochs == 0:
    BN_UNFREEZE_EPOCH = 0  # no freeze at all
else:
    BN_UNFREEZE_EPOCH = args.bn_freeze_epochs

bn_unfrozen = (BN_UNFREEZE_EPOCH == 0) or (start_epoch > BN_UNFREEZE_EPOCH)
if bn_unfrozen and BN_UNFREEZE_EPOCH > 0:
    unfreeze_bn(model)
    unfreeze_bn(wa_model)

logger.log(f'\n--- TRADES Training (epoch {start_epoch}-{NUM_ADV_EPOCHS}) ---')
logger.log(f'  batch_size={args.batch_size}, lr={args.lr}, beta={args.beta}, '
           f'scheduler={args.scheduler}, tau={args.tau}')
logger.log(f'  cutmix={args.cutmix}, label_smoothing={args.label_smoothing}')
logger.log(f'  bn_freeze={"permanent" if BN_UNFREEZE_EPOCH > NUM_ADV_EPOCHS else BN_UNFREEZE_EPOCH}')

for epoch in range(start_epoch, NUM_ADV_EPOCHS + 1):
    start_t = time.time()
    logger.log(f'\n======= Epoch {epoch} =======')
    model.train()

    # Unfreeze BN after BN_UNFREEZE_EPOCH
    if not bn_unfrozen and epoch >= BN_UNFREEZE_EPOCH:
        unfreeze_bn(model)
        unfreeze_bn(wa_model)
        # Reset BN momentum to re-estimate stats from current distribution
        set_bn_momentum(model, momentum=1.0)
        set_bn_momentum(wa_model, momentum=1.0)
        bn_unfrozen = True
        logger.log(f'  BN unfrozen at epoch {epoch}, momentum reset to 1.0')
    elif bn_unfrozen and epoch == BN_UNFREEZE_EPOCH + 1:
        # Restore normal BN momentum after one epoch of re-estimation
        set_bn_momentum(model, momentum=0.01)
        set_bn_momentum(wa_model, momentum=0.01)

    # Get base lr from FC group (scheduler controls this)
    fc_lr = args.lr
    for pg in optimizer.param_groups:
        if not pg.get('is_backbone', False):
            fc_lr = pg['lr']
            break
    last_lr = fc_lr

    # Apply backbone lr ratio (parallel-specific: sub-model backbones get scaled lr)
    ratio = backbone_lr_ratio(epoch, NUM_ADV_EPOCHS)
    for pg in optimizer.param_groups:
        if pg.get('is_backbone', False):
            pg['lr'] = fc_lr * ratio
    if epoch == start_epoch or epoch % 50 == 0:
        logger.log(f'  backbone_lr_ratio={ratio:.2f}, backbone_lr={fc_lr*ratio:.6f}, fc_lr={fc_lr:.6f}')

    update_iter = 0
    batch_metrics_all = pd.DataFrame()

    for data in tqdm(train_dataloader, desc=f'Epoch {epoch}'):
        global_step = (epoch - 1) * update_steps + update_iter
        if global_step == 0:
            set_bn_momentum(model, momentum=1.0)
        elif global_step == 1:
            set_bn_momentum(model, momentum=0.01)
        update_iter += 1

        x, y = data
        x, y = x.to(device), y.to(device)

        # Save clean x and hard labels for aux loss before CutMix
        y_hard = y.clone() if y.dim() == 1 else y.argmax(dim=1)
        x_clean = x.clone() if args.aux_weight > 0 else None

        if args.cutmix:
            cut_size = getattr(args, 'cutmix_size', None)
            x, y = cutmix(x, y, num_classes=NUM_CLASSES, cut_size=cut_size)
            y = y.to(device)

        # TRADES loss (same as WATrainer.trades_loss)
        loss, batch_metrics = trades_loss(
            model, x, y, optimizer,
            step_size=args.attack_step, epsilon=args.attack_eps,
            perturb_steps=args.attack_iter, beta=args.beta,
            attack=args.attack, label_smoothing=args.label_smoothing,
        )

        # Aux CE loss on sub-model logits (use clean x to match hard labels)
        if args.aux_weight > 0:
            aux_out = model(x_clean, return_aux=True)
            sub_logits = aux_out[:-1]
            loss = loss + aux_ce_loss(sub_logits, y_hard, class_splits, device, weight=args.aux_weight)

        loss.backward()
        if args.clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        if args.scheduler == 'cyclic':
            scheduler.step()

        # EMA update (same as WATrainer)
        global_step = (epoch - 1) * update_steps + update_iter
        ema_update(wa_model, model, global_step, decay_rate=args.tau,
                   warmup_steps=warmup_steps_ema, dynamic_decay=True)

        batch_metrics_all = pd.concat([batch_metrics_all, pd.DataFrame(batch_metrics, index=[0])],
                                       ignore_index=True)

    if args.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
        scheduler.step()

    # Update BN of WA model (same as WATrainer)
    update_bn(wa_model, model)

    # --- Logging (same structure as baseline) ---
    res = dict(batch_metrics_all.mean())
    logger.log(f'Loss: {res["loss"]:.4f}.\tLR: {last_lr:.4f}')
    if 'clean_acc' in res:
        logger.log(f'Standard Accuracy-\tTrain: {res["clean_acc"]*100:.2f}%.')
    logger.log(f'Adversarial Accuracy-\tTrain: {res.get("adversarial_acc", 0)*100:.2f}%.')

    # Test clean acc (on WA model)
    test_acc = 0.0
    wa_model.eval()
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        test_acc += accuracy(y, wa_model(x))
    test_acc /= len(test_dataloader)
    logger.log(f'Standard Accuracy-\tTest: {test_acc*100:.2f}%.')

    epoch_metrics = {'epoch': epoch, 'lr': last_lr, 'test_clean_acc': test_acc, 'test_adversarial_acc': ''}
    epoch_metrics.update({'train_' + k: v for k, v in res.items()})

    # Test adversarial acc
    if epoch % args.adv_eval_freq == 0 or epoch == NUM_ADV_EPOCHS:
        test_adv_acc = 0.0
        wa_model.eval()
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            with ctx_noparamgrad_and_eval(wa_model):
                x_adv, _ = eval_attack.perturb(x, y)
            test_adv_acc += accuracy(y, wa_model(x_adv))
        test_adv_acc /= len(test_dataloader)
        logger.log(f'Adversarial Accuracy-\tTest: {test_adv_acc*100:.2f}%.')
        epoch_metrics['test_adversarial_acc'] = test_adv_acc

    # Eval set (semi-supervised validation)
    eval_adv_acc = 0.0
    if eval_dataloader is not None:
        wa_model.eval()
        for x, y in eval_dataloader:
            x, y = x.to(device), y.to(device)
            with ctx_noparamgrad_and_eval(wa_model):
                x_adv, _ = eval_attack.perturb(x, y)
            eval_adv_acc += accuracy(y, wa_model(x_adv))
        eval_adv_acc /= len(eval_dataloader)
        logger.log(f'Adversarial Accuracy-\tEval: {eval_adv_acc*100:.2f}%.')
        epoch_metrics['eval_adversarial_acc'] = eval_adv_acc

    # Save best (same as baseline: use eval_adv if available, else test_adv)
    score = eval_adv_acc if eval_dataloader is not None else epoch_metrics.get('test_adversarial_acc', 0)
    if isinstance(score, (int, float)) and score >= old_score[1]:
        old_score = [test_acc, score]
        torch.save({
            'model_state_dict': wa_model.state_dict(),
            'unaveraged_model_state_dict': model.state_dict(),
        }, WEIGHTS_BEST)

    # Save last (for resume)
    torch.save({
        'model_state_dict': model.state_dict(),
        'wa_model_state_dict': wa_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'stage': 'adv', 'epoch': epoch, 'old_score': old_score,
    }, WEIGHTS_LAST)

    logger.log(f'Time taken: {format_time(time.time() - start_t)}')
    metrics = pd.concat([metrics, pd.DataFrame(epoch_metrics, index=[0])], ignore_index=True)
    metrics.to_csv(os.path.join(LOG_DIR, 'stats_adv.csv'), index=False)


# =========================================================
# Done
# =========================================================
logger.log(f'\nTraining completed.')
logger.log(f'Best: clean={old_score[0]*100:.2f}%, adv={old_score[1]*100:.2f}%.')
logger.log('Script Completed.')
