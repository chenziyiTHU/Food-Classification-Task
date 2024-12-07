import argparse
import os 
import torch
import torch.nn as nn
from torchvision import transforms
from models.alexnet import AlexNet, AlexNetPro
from models.vgg11 import VGG11, VGG11Pro
from models.resnet import ResNet, ResNetPro
from PIL import Image
from utils.constants import number2food, number2skin_disease


def main(args):
    model_path = args.model_path

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5478, 0.4421, 0.3401], std=[0.2254, 0.2372, 0.2352])
    ])

    if 'food' in args.model_path:
        num_classes = 10
        dataset = 'food'
    elif 'skin' in args.model_path:
        num_classes = 7
        dataset = 'skin'
    else:
        raise ValueError('Invalid dataset')
    
    if 'alexnet' in args.model_path:
        if 'pro' in args.model_path:
            model = AlexNetPro(num_classes).eval()
            model_name = 'alexnetpro'
        else:
            model = AlexNet(num_classes).eval()
            model_name = 'alexnet'
    elif 'vgg11' in args.model_path:
        if 'pro' in args.model_path:
            model = VGG11Pro(num_classes).eval()
            model_name = 'vgg11pro'
        else:
            model = VGG11(num_classes).eval()
            model_name = 'vgg11'
    elif 'resnet' in args.model_path:
        if 'pro' in args.model_path:
            model = ResNetPro(num_classes).eval()
            model_name = 'resnetpro'
        else:
            model = ResNet(num_classes).eval()
            model_name = 'resnet'
    else:
        raise ValueError('Invalid model name')
    print(f"Using model: {model_name}")
    
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
    model.eval()

    while True:
        image_path = input("Please input the image path: ")
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted = predicted.item()
        if dataset == 'food':
            print(f"Predicted food: {number2food[predicted]}")
        else:
            print(f"Predicted skin disease: {number2skin_disease[predicted]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/cnn_model.pth')
    args = parser.parse_args()
    main(args)