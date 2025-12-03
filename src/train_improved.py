import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import classification_report, f1_score, jaccard_score

# Add src to path if needed, though we are in src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import MaridaDatasetLoader, do_img_conf_mask_exist
from unet import UNet
from preprocessing import (
    normalize_image, set_low_conf_for_nan, apply_augmentations,
    build_conf_ignore_mask, apply_ignore_index_to_target
)

# --- 1. Weighted Focal Loss ---
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, ignore_index=-100):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) -> logits
        # targets: (B, H, W) -> labels
        
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # Create a mask for valid targets to avoid indexing error with ignore_index
            valid_targets = targets.clone()
            valid_targets[targets == self.ignore_index] = 0
            
            alpha_t = self.alpha[valid_targets]
            focal_loss = alpha_t * focal_loss

        # Apply ignore_index mask manually for safety
        mask = (targets != self.ignore_index).float()
        focal_loss = focal_loss * mask
        
        return focal_loss.sum() / mask.sum()

# --- 2. Post-Processing Size Filter ---
def filter_large_clusters(pred_mask, size_threshold=100, target_class=0, replacement_class=4):
    """
    Reclassifies clusters of 'target_class' larger than 'size_threshold' to 'replacement_class'.
    """
    # Create a binary mask for the target class
    binary_mask = (pred_mask == target_class).astype(int)
    
    # Label connected components
    labeled_array, num_features = label(binary_mask)
    
    # Iterate through features
    for i in range(1, num_features + 1):
        cluster_indices = (labeled_array == i)
        cluster_size = np.sum(cluster_indices)
        
        if cluster_size > size_threshold:
            pred_mask[cluster_indices] = replacement_class
            
    return pred_mask

# --- 3. Data Loading Helpers ---
def load_split_list(split_file):
    bases = []
    if not os.path.exists(split_file):
        print(f"File not found: {split_file}")
        return []
    with open(split_file, "r") as f:
        for line in f:
            name = line.strip()
            if not name: continue
            if name.endswith(".tif"): name = name[:-4]
            bases.append(name)
    return bases

def fix_base_name(base):
    # Example: "1-12-19_48MYU_0" -> "S2_1-12-19_48MYU/S2_1-12-19_48MYU_0"
    base_name = "S2_" + base
    folder_name = base_name.rsplit('_', 1)[0]
    return os.path.join(folder_name, base_name)

def get_valid_bases(split_filename, data_root):
    patches_root = os.path.join(data_root, "patches")
    splits_dir = os.path.join(data_root, "splits")
    split_path = os.path.join(splits_dir, split_filename)
    
    bases_raw = load_split_list(split_path)
    valid_bases = []
    
    for name in bases_raw:
        relative_path = fix_base_name(name)
        if do_img_conf_mask_exist(patches_root, relative_path):
            valid_bases.append(relative_path)
            
    print(f"Found {len(valid_bases)} valid patches in {split_filename}")
    return valid_bases

def train_transform(img, mask, conf):
    img, conf = set_low_conf_for_nan(img, conf, low_conf_level=3)
    img = normalize_image(img)
    img, mask, conf = apply_augmentations(img, mask, conf)
    return img, mask, conf

def val_transform(img, mask, conf):
    img, conf = set_low_conf_for_nan(img, conf, low_conf_level=3)
    img = normalize_image(img)
    return img, mask, conf

# --- 4. Main Training Script ---
# ... (Imports and classes remain the same) ...

def main():
    # Paths
    DATA_ROOT = os.path.abspath(os.path.join(os.getcwd(), "data/raw/MARIDA"))
    PATCHES_DIR = os.path.join(DATA_ROOT, "patches")
    
    # Data Loaders
    train_bases = get_valid_bases("train_X.txt", DATA_ROOT)
    val_bases = get_valid_bases("val_X.txt", DATA_ROOT)
    
    if not train_bases:
        print("No training data found. Exiting.")
        return

    train_dataset = MaridaDatasetLoader(PATCHES_DIR, train_bases, transform=train_transform)
    val_dataset = MaridaDatasetLoader(PATCHES_DIR, val_bases, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Model Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    
    model = UNet(n_channels=11, n_classes=15).to(DEVICE)
    
    # --- FIXED WEIGHTS ---
    # We bring the background (7) back up so the model doesn't paint over it.
    class_weights = torch.tensor([
        4.0,   # 0: Débris Marins
        4.0,   # 1: Sargasses D
        3.0,   # 2: Sargasses S
        3.0,   # 3: Mat. Orga
        2.0,   # 4: Navire (LOWERED to stop it eating the water)
        1.0,   # 5: Nuages
        1.0,   # 6: Eau Marine
        0.5,   # 7: Eau Sédim (RAISED: We cannot ignore the majority class!)
        4.0,   # 8: Écume 
        1.0,   # 9: Eau Trouble
        2.0,   # 10: Eau Peu Prof
        4.0,   # 11: Vagues 
        2.0,   # 12: Ombre
        5.0,   # 13: Sillage
        2.0    # 14: Mixte
    ]).to(DEVICE)
    
    criterion = WeightedFocalLoss(alpha=class_weights, gamma=2, ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Scheduler: Reduce LR if validation loss doesn't improve for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2
    )
    
    EPOCHS = 10 # Increased to allow Scheduler to work

    print("\nStarting Training...")

    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        
        for img, mask, conf in loop:
            img, mask, conf = img.to(DEVICE), mask.to(DEVICE), conf.to(DEVICE)
            
            ignore_mask = build_conf_ignore_mask(conf, threshold=2)
            target = apply_ignore_index_to_target(mask, ignore_mask, ignore_index=-100)
            
            valid = (target != -100)
            target[valid] = target[valid] - 1
            target[target == -1] = -100
            
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # --- VALIDATION (INSIDE THE LOOP) ---
        print(f"\nValidating Epoch {epoch+1}...")
        model.eval()
        val_loss = 0
        y_true_list = []
        y_pred_list = []
        
        with torch.no_grad():
            for img, mask, conf in tqdm(val_loader, desc="Val"):
                img, mask, conf = img.to(DEVICE), mask.to(DEVICE), conf.to(DEVICE)
                
                # Targets
                ignore_mask = build_conf_ignore_mask(conf, threshold=2)
                target = apply_ignore_index_to_target(mask, ignore_mask, ignore_index=-100)
                valid = (target != -100)
                target[valid] = target[valid] - 1
                target[target == -1] = -100

                # Forward
                outputs = model(img)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                
                # Predictions for Report
                preds = torch.argmax(outputs, dim=1)
                
                # Store
                valid_mask = (target != -100)
                y_true_list.append(target[valid_mask].cpu().numpy())
                y_pred_list.append(preds[valid_mask].cpu().numpy())

        # Scheduler Step
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss) # KEY: Scheduler updates here!
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr}")
        
        # --- REPORT EVERY EPOCH ---
        # (Optional: Print report only on last epoch or every 5 to save console space)
        if (epoch + 1) % 1 == 0: 
            if y_true_list:
                y_true = np.concatenate(y_true_list)
                y_pred = np.concatenate(y_pred_list)
                
                CLASS_NAMES = [
                    'Débris Marins', 'Sargasses D', 'Sargasses S', 'Mat. Orga',
                    'Navire', 'Nuages', 'Eau Marine', 'Eau Sédim',
                    'Écume', 'Eau Trouble', 'Eau Peu Prof', 'Vagues',
                    'Ombre', 'Sillage', 'Mixte'
                ]
                print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3, zero_division=0))

if __name__ == "__main__":
    main()
