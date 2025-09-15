'''Lille script til at plotte augmented data


def show_augmented_images(dataset, num_images=8):
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 4))
    for i in range(num_images):
        sample = dataset[i]
        img = sample['input']
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {sample['target']}")
        axes[i].axis('off')
    plt.show()
train_dataset = Hotdog_NotHotdog(train=True)
show_augmented_images(train_dataset)'''

#Augmenterer data i train step, ikke i test
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
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomVerticalFlip(0.3),
                transforms.RandomRotation(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
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