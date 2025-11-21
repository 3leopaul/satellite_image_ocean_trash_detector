# baseline_model.py

import torch
from torch.utils.data import DataLoader, random_split

from mock_dataset import MockDataset
from preprocessing import normalize_image, flatten_for_rf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def build_rf_dataset(
    dataloader,
    scale_factor: float = 10000.0,
    conf_threshold: int = 2,
):
    """
    Parcourt un dataloader (img, mask, conf) et construit X, y
    pour entraîner un RandomForest au niveau pixel.

    Retourne :
        X_all: np.ndarray de shape (N, C)
        y_all: np.ndarray de shape (N,)
    """
    X_list = []
    y_list = []

    for batch in dataloader:
        # batch = (image, mask, conf)
        img, mask, conf = batch

        # Le DataLoader renvoie par défaut (B, C, H, W)
        # avec B = batch_size. On prend B=1 pour simplifier.
        if img.dim() == 4:
            img = img.squeeze(0)   # (C,H,W)
            mask = mask.squeeze(0) # (H,W)
            conf = conf.squeeze(0) # (H,W)

        # 1) Normalisation (simple scaling ici)
        img_norm = normalize_image(img, scale_factor=scale_factor)

        # 2) Flatten pour RandomForest (filtre aussi par conf)
        X, y = flatten_for_rf(
            img_norm,
            mask,
            conf=conf,
            conf_threshold=conf_threshold,
        )

        # X, y sont des tensors -> on convertit en numpy
        X_list.append(X.cpu().numpy())
        y_list.append(y.cpu().numpy())

    # Concaténation de tous les patches
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    return X_all, y_all


def train_baseline_rf(
    n_patches: int = 8,
    batch_size: int = 1,
    n_estimators: int = 50,
    max_depth: int = None,
):
    """
    Entraîne un RandomForest baseline sur le MockDataset.

    n_patches : nombre de patches de MockDataset à utiliser.
    """

    # 1) On crée un dataset "faux" pour le baseline
    full_dataset = MockDataset(n=n_patches)

    # On fait un split train / val très simple (80/20)
    n_train = int(0.8 * n_patches)
    n_val = n_patches - n_train
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"Train patches: {len(train_ds)}, Val patches: {len(val_ds)}")

    # 2) Construction du dataset RF pour le train
    print("Building train RF dataset (flatten patches -> pixels)...")
    X_train, y_train = build_rf_dataset(train_loader)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # 3) Entraînement du RandomForest
    print("Training RandomForestClassifier...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        verbose=1,
    )
    rf.fit(X_train, y_train)
    print("RandomForest training done.")

    # 4) Évaluation sur le set de validation
    print("Building val RF dataset...")
    X_val, y_val = build_rf_dataset(val_loader)
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    print("Predicting on validation pixels...")
    y_pred = rf.predict(X_val)

    print("\n=== Classification report (validation pixels) ===")
    print(classification_report(y_val, y_pred, zero_division=0))

    print("\n=== Confusion matrix (validation pixels) ===")
    print(confusion_matrix(y_val, y_pred))

    return rf


if __name__ == "__main__":
    # Lancement d’un petit test avec 8 patches mock
    model = train_baseline_rf(
        n_patches=8,
        batch_size=1,
        n_estimators=50,
        max_depth=None,
    )
