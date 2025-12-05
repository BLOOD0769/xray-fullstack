# backend/data_prep.py
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class CXRFolderDataset(Dataset):
    """
    Simple folder dataset expecting structure:
      root/
        train/
          NORMAL/
          PNEUMONIA/
        val/
          NORMAL/
          PNEUMONIA/
        test/
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.samples = []
        base = os.path.join(root_dir, split)
        classes = sorted(os.listdir(base))
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        for cls in classes:
            cls_dir = os.path.join(base, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png','.jpg','.jpeg')):
                    self.samples.append((os.path.join(cls_dir,fname), self.class_to_idx[cls]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataloaders(root_dir, batch_size=16, img_size=224, num_workers=4):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = CXRFolderDataset(root_dir, 'train', transform=train_tf)
    val_ds = CXRFolderDataset(root_dir, 'val', transform=val_tf)
    test_ds = CXRFolderDataset(root_dir, 'test', transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, train_ds.class_to_idx
