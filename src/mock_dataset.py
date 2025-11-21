import torch
from torch.utils.data import Dataset

class MockDataset(Dataset):
    def __init__(self, n=8):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Fake 10-band Sentinel-like image
        image = torch.rand(10, 256, 256)
        mask = torch.randint(0, 15, (256, 256))
        conf = torch.randint(0, 4, (256, 256))
        return image, mask, conf
