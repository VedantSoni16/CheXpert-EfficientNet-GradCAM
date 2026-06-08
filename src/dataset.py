import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ChestXrayDataset(Dataset):
    """
    An optimized PyTorch Dataset class that strictly loads valid image extensions
    and converts them to Tensors for model consumption.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.image_paths = []
        self.labels = []
        
        # Define valid image extensions to prevent loading hidden system files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                # Check if the file has a valid image extension (case-insensitive)
                if img_name.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open image and force convert to RGB
        # (CheXpert images are grayscale, but pre-trained networks expect 3 channels)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)


def get_dataloaders(train_dir, val_dir, batch_size=32):
    """
    Sets up normalized image transforms and returns PyTorch DataLoaders.
    """
    # Standard ImageNet normalization parameters
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])

    # Training gets safe medical data augmentations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])

    # Validation gets zero augmentation, only structural preparation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = ChestXrayDataset(data_dir=train_dir, transform=train_transform)
    val_dataset = ChestXrayDataset(data_dir=val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_dataset.classes

if __name__ == "__main__":
    print("Dataset script updated with robust image filtering. Ready for testing.")