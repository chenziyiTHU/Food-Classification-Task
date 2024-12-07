import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.alexnet import AlexNet, AlexNetPro
from models.vgg11 import VGG11, VGG11Pro
from models.resnet import ResNet, ResNetPro
from utils.FoodDataset import FoodDataset
from utils.SkinDiseaseDataset import SkinDiseaseDataset
from utils.utils import save_config, split_dataset


def train(args):
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    if args.model_path is None:
        args.model_path = f'./checkpoints/{args.model_name}_{args.dataset}'
    model_path = args.model_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Training on device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5478, 0.4421, 0.3401], std=[0.2254, 0.2372, 0.2352]),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(45),
    ])

    if args.dataset == 'food':
        dataset = FoodDataset("data/Food-101-archive", transform=transform)
        num_classes = 10
    elif args.dataset == 'skin':
        dataset = SkinDiseaseDataset(image_dir_path="data/皮肤病数据集/HAM10000-images", label_path="data/皮肤病数据集/HAM10000-metadata.csv", transform=transform)
        num_classes = 7
    else:
        raise ValueError('Invalid dataset')
    print(f"Using dataset: {args.dataset}")

    if args.model_name == 'alexnet':
        model = AlexNet(num_classes).to(device)
    elif args.model_name == 'vgg11':
        model = VGG11(num_classes).to(device)
    elif args.model_name == 'resnet':
        model = ResNet(num_classes).to(device)
    elif args.model_name == 'alexnetpro':
        model = AlexNetPro(num_classes).to(device)
    elif args.model_name == 'vgg11pro':
        model = VGG11Pro(num_classes).to(device)
    elif args.model_name == 'resnetpro':
        model = ResNetPro(num_classes).to(device)
    else:
        raise ValueError('Invalid model name')
    print(f"Using model: {args.model_name}")
    
    if args.from_pretrained:
        print(f"Loading model from pretrained weights: {model_path}")
        model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
    else:
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
        model.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_dataset, test_dataset = split_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    print(f"Start training... Number of epochs: {num_epochs}")

    total_loss = []
    total_test_accuracy = []
    os.makedirs(model_path, exist_ok=True)
    loss_file_path = os.path.join(model_path, 'loss.txt')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_train = 0
        correct_train = 0
        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted_train = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()
        
        with torch.no_grad():
            model.eval()
            running_test_loss = 0.0
            total_test = 0
            correct_test = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                test_outputs = model(images)
                test_loss = criterion(test_outputs, labels)
                running_test_loss += test_loss.item()
                _, predicted_test = torch.max(test_outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted_test == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        test_loss = running_test_loss / len(test_loader)
        test_accuracy = correct_test / total_test

        total_loss.append(train_loss)
        total_test_accuracy.append(test_accuracy)
        print(f'Epoch [{epoch + 1}], Training loss: {train_loss}, Training accuracy: {train_accuracy}; Test loss: {test_loss}, Test accuracy: {test_accuracy}')
        with open(loss_file_path, 'a') as f:
            f.write(f'Epoch [{epoch + 1}], Training loss: {train_loss}, Training accuracy: {train_accuracy}; Test loss: {test_loss}, Test accuracy: {test_accuracy}\n')
    
    torch.save(model.state_dict(), os.path.join(model_path, 'model.pth'))
    save_config(args, os.path.join(model_path, 'config.json'))
    plt.plot(total_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(model_path, 'Training_loss.png'))
    plt.clf()
    plt.plot(total_test_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.savefig(os.path.join(model_path, 'Test_accuracy.png'))
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='alexnet')
    parser.add_argument('--dataset', type=str, default='food', choices=['food', 'skin'])
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--from_pretrained', action='store_true')
    args = parser.parse_args()
    train(args)