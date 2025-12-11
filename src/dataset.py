import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from torch.utils.data import Dataset
import rasterio

## Exploration helper functions

# Helper function that detects all base names of patches in a folder
def find_patch_bases(folder):
    bases = set()
    for f in os.listdir(folder):
        if f.endswith(".tif") and "_cl" not in f and "_conf" not in f:
            base = f.replace(".tif", "")
            bases.add(base)
    return sorted(list(bases))

# Helper function that checks if image, mask, and confidence files exist for a given base name
def do_img_conf_mask_exist(folder, base):
    img_path = os.path.join(folder, base + ".tif")
    mask_path = os.path.join(folder, base + "_cl.tif")
    conf_path = os.path.join(folder, base + "_conf.tif")
    return os.path.exists(img_path) and os.path.exists(mask_path) and os.path.exists(conf_path)

# Veirfy that all patches in a folder have image, mask, and confidence files
def check_all_patches_have_all_files(folder):
    all_good = True

    for patch_folder in os.listdir(folder):
        folder_path = os.path.join(folder, patch_folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"Checking folder: {patch_folder}")

        # List base names inside this folder
        bases = find_patch_bases(folder_path)
        print("  Found bases:", bases)

        for base in bases:
            exists = do_img_conf_mask_exist(folder_path, base)
            if exists:
                print(f"    OK: {base}")
            else:
                print(f"    MISSING FILES: {base}")
                all_good = False

    if all_good:
        print("\nAll patches have all required files.")
    else:
        print("\nSome patches are missing one or more files.")


## Dataset Loader Class

class MaridaDatasetLoader:
    def __init__(self, folder, bases, transform=None):
        self.folder = folder
        self.bases = bases
        self.transform = transform

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, idx):
        base = self.bases[idx]

        img_path = os.path.join(self.folder, base + ".tif")
        mask_path = os.path.join(self.folder, base + "_cl.tif")
        conf_path = os.path.join(self.folder, base + "_conf.tif")

        img = rasterio.open(img_path).read()          # shape (11,256,256)
        mask = rasterio.open(mask_path).read(1)       # shape (256,256)
        conf = rasterio.open(conf_path).read(1)       # shape (256,256)

        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)
        conf = torch.tensor(conf, dtype=torch.long)

        if self.transform:
            img, mask, conf = self.transform(img, mask, conf) 

        return img, mask, conf