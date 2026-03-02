import torch
import torch.nn.functional as F


def cutmix(images, labels, num_classes=10):
    """
    CutMix with patch area ratio sampled from Beta(1,1) = Uniform(0,1)
    This matches the Rebuffi et al. NeurIPS 2021 Table 1 results.
    """
    batch_size, _, height, width = images.shape

    # Sample lambda from Beta(1,1) = Uniform(0,1)
    lam = torch.distributions.Beta(1.0, 1.0).sample().item()

    # Compute window size from lambda
    cut_ratio = (1.0 - lam) ** 0.5
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
