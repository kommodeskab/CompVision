from torch.utils.data import Dataset
import os
import glob
from PIL import Image
from torchvision import transforms
from torch import Tensor
from dataclasses import dataclass

@dataclass
class BaseBatch:
    input: Tensor
    target: Tensor

class Hotdog_NotHotdog(Dataset):
    def __init__(self, train : bool, image_size : int = 32):
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
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return {
            'input': X,
            'target': y,
        }