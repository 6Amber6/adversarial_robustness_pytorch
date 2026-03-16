import torch
import torch.nn.functional as F


def cutmix(images, labels, num_classes, cut_size=None):
    """
    CutMix augmentation.
    If cut_size is set: fixed window (Rebuffi et al. NeurIPS 2021, optimal=20 for 32x32).
    If cut_size is None: Beta(1,1) sampling.
    num_classes: must be passed (10 for CIFAR-10, 100 for CIFAR-100).
    """
    batch_size, _, height, width = images.shape

    if cut_size is not None:
        # Fixed window (optimal window length=20 for CIFAR-10 32x32)
        cut_h = min(cut_size, height)
        cut_w = min(cut_size, width)
    else:
        # Beta(1,1): lam = patch area ratio, cut_ratio = lam^0.5 = side length ratio
        lam = torch.distributions.Beta(1.0, 1.0).sample().item()
        cut_ratio = lam ** 0.5
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)

    # Random center
    cx = torch.randint(0, width, (1,)).item()
    cy = torch.randint(0, height, (1,)).item()

    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, height)

    # Shuffle batch
    perm = torch.randperm(batch_size, device=images.device)
    images_mix = images.clone()
    images_mix[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]

    # Recompute actual lambda (pixel ratio)
    actual_lam = 1.0 - (x2 - x1) * (y2 - y1) / (height * width)

    # Soft labels (defensive: labels may already be one-hot if CutMix called multiple times)
    if labels.dim() == 1:
        labels_onehot = F.one_hot(labels.long(), num_classes).float()
        labels_perm = F.one_hot(labels[perm].long(), num_classes).float()
    else:
        labels_onehot = labels.float()
        labels_perm = labels[perm].float()
    labels_mix = actual_lam * labels_onehot + (1.0 - actual_lam) * labels_perm

    return images_mix, labels_mix
