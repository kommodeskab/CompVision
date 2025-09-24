from torch.utils.data import Dataset
import os
import glob
from PIL import Image
from torchvision import transforms
from torchvision.transforms.v2 import GaussianNoise

class Hotdog_NotHotdog(Dataset):
    def __init__(self, train : bool, image_size : int = 32):
        if train:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=(-0.2, 0.2)),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomRotation(degrees=20),
                transforms.RandomResizedCrop((image_size, image_size), scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.2),
                transforms.ToTensor(),
                GaussianNoise(0., 0.001),
                transforms.RandomErasing(p=0.2),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        data_path = os.path.join('hotdog_nothotdog', 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        X = (2 * X - 1).clamp(-1, 1)
        return {
            'input': X,
            'target': y,
        }