# baseline_model.py

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import (
    normalize_image,
    flatten_for_rf,
    set_low_conf_for_nan,
)


def build_rf_dataset(
    dataloader,
    conf_threshold: int = 2,
):
    """
    Parcourt un dataloader (img, mask, conf) et construit X, y
    pour entraîner un RandomForest au niveau pixel.

    Args:
        dataloader: DataLoader qui renvoie (img, mask, conf)
        conf_threshold: on garde les pixels avec conf >= threshold

    Retourne :
        X_all: np.ndarray de shape (N, C)
        y_all: np.ndarray de shape (N,)
    """
    X_list = []
    y_list = []

    for batch in dataloader:
        img, mask, conf = batch

        # (B,C,H,W) -> (C,H,W) si batch_size=1
        if img.dim() == 4:
            img = img.squeeze(0)
            mask = mask.squeeze(0)
            conf = conf.squeeze(0)

        # 1) Normalisation + nettoyage NaN/Inf
        img = normalize_image(img)
        img, conf = set_low_conf_for_nan(img, conf, low_conf_level=3)

        # 2) Flatten pour RandomForest (filtre aussi par conf)
        X, y = flatten_for_rf(
            img,
            mask,
            conf=conf,
            conf_threshold=conf_threshold,
        )

        X_list.append(X.cpu().numpy())
        y_list.append(y.cpu().numpy())

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    return X_all, y_all


def train_baseline_rf(
    train_loader,
    val_loader,
    n_estimators: int = 50,
    max_depth: int | None = None,
):
    """
    Entraîne un RandomForest baseline à partir de DataLoader déjà construits.

    - train_loader / val_loader doivent renvoyer des batches (img, mask, conf).
    - Les splits / chemins sont gérés ailleurs (ex: dans notebook_main.ipynb).
    """

    print(f"Train patches: {len(train_loader.dataset)}, Val patches: {len(val_loader.dataset)}")

    # 1) Construction du dataset RF pour le train
    print("Building train RF dataset (flatten patches -> pixels)...")
    X_train, y_train = build_rf_dataset(train_loader)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # 2) Entraînement du RandomForest
    print("Training RandomForestClassifier...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        verbose=1,
    )
    rf.fit(X_train, y_train)
    print("RandomForest training done.")

    # 3) Évaluation sur le set de validation
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
