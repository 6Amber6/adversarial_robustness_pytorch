"""
Parallel Fusion WideResNet with Swish activation.

Sub-models are trained on different class subsets, then their embeddings
are concatenated and fused through a linear layer for full-class prediction.
Based on the Swish WRN architecture from Gowal et al., 2020.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wideresnetwithswish import WideResNet, _ACTIVATION, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD


class WRNWithEmbeddingSwish(WideResNet):
    """WideResNet+Swish that can return embeddings before the final FC layer."""

    def forward(self, x, return_embedding=False):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.std_cuda = self.std.cuda()
            out = (x - self.mean_cuda) / self.std_cuda
        else:
            out = (x - self.mean) / self.std

        out = self.init_conv(out)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        emb = out.view(out.size(0), -1)
        logits = self.logits(emb)
        if return_embedding:
            return emb, logits
        return logits


class ParallelFusionWRNSwish(nn.Module):
    """
    Parallel concat fusion: sub-models produce embeddings,
    concatenated and projected via a linear layer to full-class logits.

    Sub-model parameters are frozen in __init__; the training script
    unfreezes them explicitly before Stage 2.
    """

    def __init__(self, sub_models: List[WRNWithEmbeddingSwish], num_classes: int = 10):
        super().__init__()
        self.subs = nn.ModuleList(sub_models)
        self.num_subs = len(sub_models)

        for sub in self.subs:
            for p in sub.parameters():
                p.requires_grad = False

        total_emb_dim = sum(s.num_channels for s in sub_models)
        self.fc = nn.Linear(total_emb_dim, num_classes)

    def forward(self, x, return_aux=False):
        embs, sub_logits = [], []
        for sub in self.subs:
            emb, logits = sub(x, return_embedding=True)
            embs.append(emb)
            sub_logits.append(logits)

        out = self.fc(torch.cat(embs, dim=1))
        if return_aux:
            return sub_logits + [out]
        return out


# ==================== Class splits ====================

# CIFAR-10: semantic split (vehicle vs animal)
CIFAR10_SPLITS = {
    'vehicle': [0, 1, 8, 9],           # plane, car, ship, truck
    'animal':  [2, 3, 4, 5, 6, 7],     # bird, cat, deer, dog, frog, horse
}

# CIFAR-100 fine->coarse mapping (100 fine classes -> 20 superclasses)
# 0 aquatic_mammals, 1 fish, 2 flowers, 3 food_containers, 4 fruit_veg,
# 5 household_electrical, 6 household_furniture, 7 insects, 8 large_carnivores,
# 9 large_manmade_outdoor, 10 large_natural_outdoor, 11 large_herbivores,
# 12 medium_mammals, 13 non-insect_invertebrates, 14 people, 15 reptiles,
# 16 small_mammals, 17 trees, 18 vehicles_1, 19 vehicles_2
CIFAR100_FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]

def _coarse_to_fine(coarse_list):
    """Convert list of coarse class IDs to sorted list of fine class IDs."""
    return sorted([i for i, c in enumerate(CIFAR100_FINE_TO_COARSE) if c in coarse_list])

# CIFAR-100 2-group: Nature/Biology vs Man-made/Scenes (semantic, 10 superclasses each)
_COARSE_2G = {
    'nature':  [0, 1, 2, 7, 8, 11, 12, 13, 15, 16],
    # aquatic_mammals, fish, flowers, insects, large_carnivores,
    # large_herbivores, medium_mammals, non-insect_invertebrates, reptiles, small_mammals
    'manmade': [3, 4, 5, 6, 9, 10, 14, 17, 18, 19],
    # food_containers, fruit_veg, household_electrical, household_furniture,
    # large_manmade_outdoor, large_natural_outdoor, people, trees, vehicles_1, vehicles_2
}
CIFAR100_2x50_SPLITS = {name: _coarse_to_fine(cs) for name, cs in _COARSE_2G.items()}

# CIFAR-100 4-group: Textural/semantic categories (5 superclasses each)
_COARSE_4G = {
    'textured_organic': [7, 15, 16, 8, 13],
    # insects, reptiles, small_mammals, large_carnivores, non-insect_invertebrates
    'smooth_organic':   [1, 0, 11, 12, 14],
    # fish, aquatic_mammals, large_herbivores, medium_mammals, people
    'rigid_manmade':    [6, 5, 3, 18, 9],
    # household_furniture, household_electrical, food_containers, vehicles_1, large_manmade_outdoor
    'large_structures': [10, 17, 19, 2, 4],
    # large_natural_outdoor, trees, vehicles_2, flowers, fruit_veg
}
CIFAR100_4x25_SPLITS = {name: _coarse_to_fine(cs) for name, cs in _COARSE_4G.items()}


def get_class_splits(dataset: str, num_groups: int = 2) -> dict:
    if 'cifar100' in dataset:
        if num_groups == 4:
            return CIFAR100_4x25_SPLITS
        return CIFAR100_2x50_SPLITS
    return CIFAR10_SPLITS


def create_parallel_fusion(depth: int = 28, width: int = 10, act_fn: str = 'swish',
                           dataset: str = 'cifar10', num_classes: int = 10,
                           class_splits: Optional[dict] = None):
    """
    Create ParallelFusionWRNSwish with sub-models for each class split.
    Returns (fusion_model, class_splits_dict).
    """
    if class_splits is None:
        class_splits = get_class_splits(dataset)

    mean, std = (CIFAR100_MEAN, CIFAR100_STD) if 'cifar100' in dataset else (CIFAR10_MEAN, CIFAR10_STD)
    activation = _ACTIVATION[act_fn]

    sub_models = []
    for name, classes in class_splits.items():
        sub = WRNWithEmbeddingSwish(
            num_classes=len(classes), depth=depth, width=width,
            activation_fn=activation, mean=mean, std=std,
        )
        sub_models.append(sub)

    return ParallelFusionWRNSwish(sub_models, num_classes=num_classes), class_splits
