import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FoodDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        label = self.labels[idx]
        return image, label


def load_data(data_dir):
    image_paths = []
    labels = []
    
    for class_name in ['food', 'non_food']:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, filename))
                    labels.append(1 if class_name == 'food' else 0)
    
    return image_paths, labels


def get_transforms():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.HueSaturationValue(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform


def create_data_loaders(data_dir, batch_size=32, test_size=0.2, random_state=42):
    image_paths, labels = load_data(data_dir)
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = FoodDataset(train_paths, train_labels, train_transform)
    val_dataset = FoodDataset(val_paths, val_labels, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def augment_data(data_dir, output_dir, augment_factor=2):
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in ['food', 'non_food']:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    augment_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.RandomCrop(224, 224, p=0.5),
        A.ElasticTransform(p=0.3),
        A.GridDistortion(p=0.3),
    ])
    
    for class_name in ['food', 'non_food']:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_dir, filename)
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    base_name = os.path.splitext(filename)[0]
                    ext = os.path.splitext(filename)[1]
                    
                    original_path = os.path.join(output_dir, class_name, f"{base_name}_orig{ext}")
                    cv2.imwrite(original_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    
                    for i in range(augment_factor):
                        augmented = augment_transform(image=image)['image']
                        aug_path = os.path.join(output_dir, class_name, f"{base_name}_aug_{i}{ext}")
                        cv2.imwrite(aug_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
