import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm

from models.alexnet import AlexNet, AlexNetPro
from models.vgg11 import VGG11, VGG11Pro
from models.resnet import ResNet, ResNetPro
from utils.FoodDataset import FoodDataset
from utils.SkinDiseaseDataset import SkinDiseaseDataset


def eval_model(args):
    model_path = args.model_path
    batch_size = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Evaluation on device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5478, 0.4421, 0.3401], std=[0.2254, 0.2372, 0.2352])
    ])

    if 'food' in args.model_path:
        num_classes = 10
        dataset = FoodDataset("data/Food-101-archive", transform=transform)
    elif 'skin' in args.model_path:
        num_classes = 7
        dataset = SkinDiseaseDataset(image_dir_path="data/皮肤病数据集/HAM10000-images", label_path="data/皮肤病数据集/HAM10000-metadata.csv", transform=transform)
    print(f"Evaluate on dataset: {args.dataset}")

    if 'alexnet' in args.model_path:
        if 'pro' in args.model_path:
            model = AlexNetPro(num_classes).eval()
            model_name = 'alexnetpro'
        else:
            model = AlexNet(num_classes).eval()
            model_name = 'alexnet'
        target_layers = [model.features[-4]]
    elif 'vgg11' in args.model_path:
        if 'pro' in args.model_path:
            model = VGG11Pro(num_classes).eval()
            model_name = 'vgg11pro'
        else:
            model = VGG11(num_classes).eval()
            model_name = 'vgg11'
        target_layers = [model.features[-4]]
    elif 'resnet' in args.model_path:
        if 'pro' in args.model_path:
            model = ResNetPro(num_classes).eval()
            model_name = 'resnetpro'
        else:
            model = ResNet(num_classes).eval()
            model_name = 'resnet'
        target_layers = [model.layer4[-1].conv2]
    else:
        raise ValueError('Invalid model name')
    
    print(f"Using model: {args.model_name}")
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
    model.eval()
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0
    print("Start evaluating...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Loss: {running_loss / len(test_loader)}, Accuracy: {correct / total}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    eval_model(args)