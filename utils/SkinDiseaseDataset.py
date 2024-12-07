import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.constants import skin_disease2number, number2skin_disease


class SkinDiseaseDataset(Dataset):
    def __init__(
        self,
        image_dir_path="data/皮肤病数据集/HAM10000-images",
        label_path="data/皮肤病数据集/HAM10000-metadata.csv",
        transform=None
    ):
        annotations = pd.read_csv(label_path)
        self.image_paths = []
        self.labels = []
        self.transform = transform
        for row in range(len(annotations)):
            image_id = annotations.iloc[row, 0]
            image_path = os.path.join(image_dir_path, image_id + ".jpg")
            self.image_paths.append(image_path)
            for i in range(1, len(annotations.iloc[0])):
                if annotations.iloc[row, i] == 1.0:
                    self.labels.append(skin_disease2number[annotations.columns[i]])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = SkinDiseaseDataset(transform=transform)
    for sample in dataset:
        print(sample)