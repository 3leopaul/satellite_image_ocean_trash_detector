# Ce fichier définit les fonctions de prétraitement suivantes :
#   - normalize_image
#   - augmentations simples (flip, rotation)
#   - masque d’ignorance basé sur la confiance
#   - flatten_for_rf (mise à plat pour RandomForest)
# Et à la fin un petit TEST utilisant des données factices.

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from mock_dataset import MockDataset
from typing import Tuple, Optional
import torch
import torch.nn.functional as F


# 1. Normalisation + traitement des NaN
def normalize_image(img: torch.Tensor) -> torch.Tensor:
    """
    Normalise un patch MARIDA en appliquant :
    1) Un clipping des valeurs physiques dans [0, 1]
       (corrige les valeurs légèrement négatives ou >1 dues à la correction atmosphérique)
    2) Une normalisation min-max par bande pour ramener chaque bande dans [0, 1]

    Cette normalisation est adaptée aux modèles U-Net et stabilise fortement l'entraînement.
    """

    img = img.float()

    # Étape 1 : Clamping dans la plage [0,1]
    img = torch.clamp(img, 0.0, 1.0)

    # Étape 2 : Normalisation min-max par bande
    # img a la forme (C, H, W)
    bands_min = img.amin(dim=(1, 2), keepdim=True)
    bands_max = img.amax(dim=(1, 2), keepdim=True)

    # éviter une division par zéro
    denom = (bands_max - bands_min).clamp(min=1e-6)

    img = (img - bands_min) / denom

    return img




def compute_dataset_stats(dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calcule la moyenne et l’écart-type par bande sur un dataloader d’images.
    Suppose que le dataloader renvoie (image, mask, conf) où :
        image : (B, C, H, W) ou (C, H, W)
    """
    n_pixels = 0
    sum_ = None
    sum_sq = None

    for batch in dataloader:
        # batch peut être (img, mask, conf) ou un dict
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        elif isinstance(batch, dict):
            imgs = batch["image"]
        else:
            raise ValueError("Type de batch non supporté pour le calcul des statistiques.")

        if imgs.dim() == 3:  # (C,H,W) -> (1,C,H,W)
            imgs = imgs.unsqueeze(0)

        imgs = imgs.float()
        B, C, H, W = imgs.shape
        pixels_in_batch = B * H * W

        imgs_flat = imgs.view(B, C, -1)  # (B, C, H*W)
        batch_sum = imgs_flat.sum(dim=(0, 2))            # (C,)
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



def set_low_conf_for_nan(
    img: torch.Tensor,
    conf: torch.Tensor,
    low_conf_level: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Si un pixel contient un NaN dans AU MOINS une bande,
    on définit sa confiance au niveau faible (3).

    Args:
        img:  (C, H, W)
        conf: (H, W)
    Returns:
        img_clean: (C, H, W) sans NaN/Inf
        conf_new:  (H, W) avec pixels invalides marqués comme faible confiance
    """
    # True là où une bande = NaN ou Inf
    invalid = ~torch.isfinite(img)
    invalid_per_pixel = invalid.any(dim=0)

    conf_new = conf.clone()
    conf_new[invalid_per_pixel] = low_conf_level

    # Remplacement des NaN/Inf pour éviter les explosions numériques
    img_clean = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    return img_clean, conf_new



# 2. AUGMENTATIONS (flip + rotation)

def apply_augmentations(
    img: torch.Tensor,
    mask: torch.Tensor,
    conf: Optional[torch.Tensor] = None,
    p_hflip: float = 0.5,
    p_vflip: float = 0.5,
    p_rotate90: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Applique des augmentations de manière cohérente sur img, mask, et conf.
    Args:
        img:  (C, H, W)
        mask: (H, W)
        conf: (H, W) ou None
    """

    # Flip horizontal
    if torch.rand(1).item() < p_hflip:
        img = torch.flip(img, dims=[2])
        mask = torch.flip(mask, dims=[1])
        if conf is not None:
            conf = torch.flip(conf, dims=[1])

    # Flip vertical
    if torch.rand(1).item() < p_vflip:
        img = torch.flip(img, dims=[1])
        mask = torch.flip(mask, dims=[0])
        if conf is not None:
            conf = torch.flip(conf, dims=[0])

    # Rotation aléatoire de k * 90°
    if torch.rand(1).item() < p_rotate90:
        k = torch.randint(low=0, high=4, size=(1,)).item()
        if k > 0:
            img = torch.rot90(img, k=k, dims=(1, 2))
            mask = torch.rot90(mask, k=k, dims=(0, 1))
            if conf is not None:
                conf = torch.rot90(conf, k=k, dims=(0, 1))

    return img, mask, conf



# 3. GESTION DE LA CONFIANCE (IGNORE MASK)

def build_conf_ignore_mask(
    conf: torch.Tensor,
    threshold: int = 2,
) -> torch.Tensor:
    """
    Construit un masque booléen où les pixels avec confiance > threshold sont ignorés.
    Confiance :
        1 : Haute → garder
        2 : Modérée → garder
        3 : Faible → ignorer
    Args:
        conf: (H, W)
        threshold: garde les pixels avec conf <= threshold.
    Returns:
        ignore_mask: (H, W) bool, True = IGNORER le pixel.
    """
    ignore_mask = conf > threshold
    return ignore_mask



def apply_ignore_index_to_target(
    target: torch.Tensor,
    ignore_mask: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Remplace dans la cible les pixels à ignorer par ignore_index.
    Args:
        target: (H, W) labels de classes
        ignore_mask: (H, W) bool
        ignore_index: valeur spéciale utilisée dans la loss
    Returns:
        target_mod: (H, W)
    """
    target_mod = target.clone()
    target_mod[ignore_mask] = ignore_index
    return target_mod






# 5. Petit test avec données factices

if __name__ == "__main__":
    print("\n\n Test factice - preprocessing.py \n")

    # Faux patch MARIDA
    C, H, W = 10, 256, 256
    img = torch.randint(0, 10000, (C, H, W))  # valeurs DN style Sentinel-2 brut
    mask = torch.randint(0, 15, (H, W))       # 15 classes
    conf = torch.randint(1, 4, (H, W))        # niveaux de confiance 1..3

    print(f"Shape image originale : {img.shape}")
    print(f"Shape mask original : {mask.shape}")
    print(f"Shape conf originale : {conf.shape}")

    # 1) Test normalisation
    img_norm = normalize_image(img)
    print(f"Après normalize_image → min={img_norm.min():.4f}, max={img_norm.max():.4f}")

    # 2) Test augmentations
    img_aug, mask_aug, conf_aug = apply_augmentations(img_norm, mask, conf)
    print(f"Après augmentations → img: {img_aug.shape}, mask: {mask_aug.shape}, conf: {conf_aug.shape}")

    # 3) Test du ignore_mask basé sur confiance
    ignore_mask = build_conf_ignore_mask(conf_aug, threshold=2)
    mask_ignored = apply_ignore_index_to_target(mask_aug, ignore_mask, ignore_index=-100)
    print(f"Pixels ignorés (True) : {ignore_mask.sum().item()}")
    print(f"Valeurs uniques dans mask_ignored (extrait) : {torch.unique(mask_ignored)[:10]}")


    print("\n Test factice terminé !")
