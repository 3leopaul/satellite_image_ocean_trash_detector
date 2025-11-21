# This is the preprocessing files it defines:
#   - normalize_image
#   - simple augmentations (flip, rotate)
#   - confidence-based ignore mask
#   - flatten_for_rf
# And at the bottom there is a small TEST using fake data.

from mock_dataset import MockDataset
from typing import Tuple, Optional
import torch
import torch.nn.functional as F


# 1. First we have to normalize


def normalize_image(img: torch.Tensor, scale_factor: float = 10000.0, mean: Optional[torch.Tensor] = None, 
                    std: Optional[torch.Tensor] = None,
                    ) -> torch.Tensor:
    """
    Normalize a Sentinel image tensor
    Args:
        img:  Tensor (C, H, W). Can be int or float. C: Channels, H: Height, W: Width
        scale_factor: if > 0, divide by this first (Sentinel-2 usually /10000).
        mean: optional, band means. If None, only scaling is applied.
        std:  optional, band std. If None, only scaling is applied.
    Returns:
        img_norm: Tensor (C, H, W), float32.
    """
    if not torch.is_floating_point(img):
        img = img.float()

    if scale_factor is not None and scale_factor > 0:
        img = img / scale_factor

    # If no mean/std given, just return scaled image
    if mean is None or std is None:
        return img

    # We make sure mean/std are tensors on same device
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, dtype=img.dtype, device=img.device)
    if not torch.is_tensor(std):
        std = torch.tensor(std, dtype=img.dtype, device=img.device)

    std = torch.clamp(std, min=1e-6)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)

    img = (img - mean) / std
    return img


def compute_dataset_stats(dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-band mean and std over a dataloader of images.
    Assumes dataloader yields (image, mask, conf) where:
        image: (B, C, H, W) or (C, H, W) - B: Batch size
    """
    n_pixels = 0
    sum_ = None
    sum_sq = None

    for batch in dataloader:
        # batch can be (img, mask, conf) or dict
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        elif isinstance(batch, dict):
            imgs = batch["image"]
        else:
            raise ValueError("Unsupported batch type for stats computation")

        if imgs.dim() == 3:  # (C,H,W) -> (1,C,H,W)
            imgs = imgs.unsqueeze(0)

        imgs = imgs.float()  # (B,C,H,W)
        B, C, H, W = imgs.shape
        pixels_in_batch = B * H * W

        imgs_flat = imgs.view(B, C, -1)  # (B, C, H*W)
        batch_sum = imgs_flat.sum(dim=(0, 2))      # (C,)
        batch_sum_sq = (imgs_flat ** 2).sum(dim=(0, 2))  # (C,)

        if sum_ is None:
            sum_ = batch_sum
            sum_sq = batch_sum_sq
        else:
            sum_ += batch_sum
            sum_sq += batch_sum_sq

        n_pixels += pixels_in_batch

    mean = sum_ / n_pixels
    var = (sum_sq / n_pixels) - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-6))

    return mean, std


# 2. AUGMENTATIONS (flip + rotate)

def apply_augmentations( img: torch.Tensor, mask: torch.Tensor, conf: Optional[torch.Tensor] = None, p_hflip: float = 0.5,
    p_vflip: float = 0.5, p_rotate90: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    augmentations applied consistently on img, mask, conf.
    Args:
        img:  (C, H, W)
        mask: (H, W)
        conf: (H, W) or None
    """
    # Horizontal flip (left-right)
    if torch.rand(1).item() < p_hflip:
        img = torch.flip(img, dims=[2])   # flip width axis
        mask = torch.flip(mask, dims=[1])
        if conf is not None:
            conf = torch.flip(conf, dims=[1])

    # Vertical flip (up-down)
    if torch.rand(1).item() < p_vflip:
        img = torch.flip(img, dims=[1])   # flip height axis
        mask = torch.flip(mask, dims=[0])
        if conf is not None:
            conf = torch.flip(conf, dims=[0])

    # Random rotation by k * 90 degrees
    if torch.rand(1).item() < p_rotate90:
        k = torch.randint(low=0, high=4, size=(1,)).item()  # 0,1,2,3
        if k > 0:
            img = torch.rot90(img, k=k, dims=(1, 2))
            mask = torch.rot90(mask, k=k, dims=(0, 1))
            if conf is not None:
                conf = torch.rot90(conf, k=k, dims=(0, 1))

    return img, mask, conf


# 3. CONFIDENCE HANDLING


def build_conf_ignore_mask(conf: torch.Tensor, threshold: int = 2,
    ) -> torch.Tensor:
    """
    Build a boolean mask where pixels with confidence > threshold are ignored.
    1: High - keep
    2: Moderate - keep
    3: Low - we ignore
    Args:
        conf: (H, W) integer confidence levels.
        threshold: keep pixels with conf <= threshold.
    Returns:
        ignore_mask: (H, W) bool tensor, True = IGNORE this pixel.
    """
    ignore_mask = conf > threshold
    return ignore_mask


def apply_ignore_index_to_target( target: torch.Tensor, ignore_mask: torch.Tensor, ignore_index: int = -100,
    ) -> torch.Tensor:
    """
    The function sets target pixels to ignore_index where ignore_mask is True.
    Args:
        target: (H, W) class ids.
        ignore_mask: (H, W) bool.
        ignore_index: value used in loss 
    Returns:
        target_mod: (H, W) with some pixels set to ignore_index.
    """
    target_mod = target.clone()
    target_mod[ignore_mask] = ignore_index
    return target_mod


# 4. FLATTEN FOR RANDOM FOREST


def flatten_for_rf( img: torch.Tensor, mask: torch.Tensor, conf: Optional[torch.Tensor] = None, conf_threshold: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten image + mask into pixel-level examples for RandomForest.
    Args:
        img:  (C, H, W)
        mask: (H, W)
        conf: (H, W) or None
        conf_threshold: keep pixels with conf <= threshold.
    Returns:
        X: (N, C) tensor of features (per-pixel bands)
        y: (N,) tensor of labels
    """
    C, H, W = img.shape
    img = img.float()
    X = img.view(C, -1).T          # (H*W, C)
    y = mask.view(-1)              # (H*W,)

    if conf is not None:
        conf_flat = conf.view(-1)
        keep = conf_flat <= conf_threshold
        X = X[keep]
        y = y[keep]

    return X, y



# 5. Small fake test


if __name__ == "__main__":
    print("\n\n Fake test - preprocessing.py \n")

    # Fake MARIDA patch
    C, H, W = 10, 256, 256
    img = torch.randint(0, 10000, (C, H, W))  # like raw Sentinel-2 DN values
    mask = torch.randint(0, 15, (H, W))       # 15 classes
    conf = torch.randint(1, 4, (H, W))        # confidence 0..3

    print(f"Original img shape: {img.shape}")
    print(f"Original mask shape: {mask.shape}")
    print(f"Original conf shape: {conf.shape}")

    # 1) Test normalization
    img_norm = normalize_image(img, scale_factor=10000.0)
    print(f"After normalize_image -> min={img_norm.min():.4f}, max={img_norm.max():.4f}")

    # 2) Test augmentations
    img_aug, mask_aug, conf_aug = apply_augmentations(img_norm, mask, conf)
    print(f"After augmentations -> img: {img_aug.shape}, mask: {mask_aug.shape}, conf: {conf_aug.shape}")

    # 3) Test confidence mask
    ignore_mask = build_conf_ignore_mask(conf_aug, threshold=2)
    mask_ignored = apply_ignore_index_to_target(mask_aug, ignore_mask, ignore_index=-100)
    print(f"Ignore_mask true count: {ignore_mask.sum().item()} pixels")
    print(f"mask_ignored unique values (sample): {torch.unique(mask_ignored)[:10]}")

    # 4) Test flatten_for_rf
    X_rf, y_rf = flatten_for_rf(img_aug, mask_aug, conf_aug, conf_threshold=2)
    print(f"RandomForest features shape: X={X_rf.shape}, y={y_rf.shape}")

    print("\n Fake test finished !!!")