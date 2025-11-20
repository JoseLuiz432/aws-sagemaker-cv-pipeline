import os
import torch
from torchvision import transforms, datasets

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224
RESIZE_VAL = 256

def get_transforms(is_training=True):
    """
    Retorna o pipeline de transformação correto para cada fase.
    """
    if is_training:
        # Data Augmentation para treino
        return transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        # Apenas redimensionamento e normalização para Validação/Inferência
        return transforms.Compose([
            transforms.Resize(RESIZE_VAL),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

def create_dataloaders(data_dir, batch_size, num_workers=2):
    """
    Cria os DataLoaders do PyTorch lendo do disco.
    """
    dirs = {'train': os.path.join(data_dir, 'train'),
            'val': os.path.join(data_dir, 'val')}
    
    # Dicionário de datasets aplicando os transforms corretos
    image_datasets = {
        'train': datasets.ImageFolder(dirs['train'], get_transforms(is_training=True)),
        'val': datasets.ImageFolder(dirs['val'], get_transforms(is_training=False))
    }
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
        
    return dataloaders