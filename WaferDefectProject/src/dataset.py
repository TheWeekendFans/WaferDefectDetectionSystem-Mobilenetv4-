import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class WaferMapDataset(Dataset):
    def __init__(self, data_path, mode='train', test_size=0.2, transform=None):
        """
        Args:
            data_path (str): Path to .npz file
            mode (str): 'train' or 'test'
            test_size (float): Ratio of test set
            transform (callable, optional): Transform to apply to sample
        """
        try:
            data = np.load(data_path, allow_pickle=True)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load data from {data_path}: {e}")
            
        self.images = data['arr_0']
        self.labels = data['arr_1']
        self.transform = transform
        
        # Label decoding: map unique binary rows to integers 0-37
        # key: tuple(binary_vector), value: int_class_index
        labels_as_tuples = [tuple(row) for row in self.labels]
        self.unique_labels = sorted(list(set(labels_as_tuples)))
        self.label_map = {label: i for i, label in enumerate(self.unique_labels)}
        self.num_classes = len(self.unique_labels)
        
        self.targets = np.array([self.label_map[l] for l in labels_as_tuples])
        
        # Train/Test Split
        indices = np.arange(len(self.images))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split_idx = int(len(self.images) * (1 - test_size))
        
        if mode == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
            
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        img_arr = self.images[real_idx] # Shape (52, 52)
        target = self.targets[real_idx]
        
        # Simple scaling for visualization/feature extraction
        # Original values roughly 0-3. Map to 0-255.
        img_arr = (img_arr * (255.0 / 3.0)).astype(np.uint8)
        
        # Convert to RGB
        img = Image.fromarray(img_arr, mode='L').convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(target, dtype=torch.long)

if __name__ == '__main__':
    data_path = r'g:/半导体offer/dataset_repo/Wafer_Map_Datasets.npz'
    ds = WaferMapDataset(data_path, mode='train')
    print(f"Dataset Loaded. Number of classes: {ds.num_classes}")
    print(f"Train samples: {len(ds)}")
    
    img, label = ds[0]
    print(f"Sample 0 shape: {img.size} (W,H), Label: {label}")
