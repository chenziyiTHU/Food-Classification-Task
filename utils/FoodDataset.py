import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.constants import food2number, number2food


class FoodDataset(Dataset):
    def __init__(
        self,
        image_dir_path="data/Food-101-archive",
        transform=None
    ):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        for dirpath, dirnames, filenames in os.walk(image_dir_path):
            for dirname in dirnames:
                if dirname != '.' and dirname != '..':
                    for _, _, files in os.walk(os.path.join(dirpath, dirname)):
                        for file in files:
                            self.image_paths.append(os.path.join(dirpath, dirname, file))
                            self.labels.append(food2number[dirname])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
    
    
def cal_mean_std(loader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images_count = 0
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    return mean, std
    

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = FoodDataset(transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    mean, std = cal_mean_std(loader)
    print(mean, std)