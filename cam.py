import torch
from torch import nn
import numpy as np
import cv2
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.alexnet import AlexNet, AlexNetPro
from models.vgg11 import VGG11, VGG11Pro
from models.resnet import ResNet, ResNetPro
from utils.constants import number2food

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--target_class', type=int, default=0)
    args = parser.parse_args()

    if 'food' in args.model_path:
        num_classes = 10
    elif 'skin' in args.model_path:
        num_classes = 7

    if 'alexnet' in args.model_path:
        if 'pro' in args.model_path:
            model = AlexNetPro(num_classes).eval()
            model_name = 'alexnetpro'
            target_layers = [model.features[-4]]
        else:
            model = AlexNet(num_classes).eval()
            model_name = 'alexnet'
            target_layers = [model.features[-3]]
    elif 'vgg11' in args.model_path:
        if 'pro' in args.model_path:
            model = VGG11Pro(num_classes).eval()
            model_name = 'vgg11pro'
            target_layers = [model.features[-4]]
        else:
            model = VGG11(num_classes).eval()
            model_name = 'vgg11'
            target_layers = [model.features[-3]]
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
    
    model.load_state_dict(torch.load(os.path.join(args.model_path, 'model.pth')))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5478, 0.4421, 0.3401], std=[0.2254, 0.2372, 0.2352])
    ])

    image = Image.open(args.image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    image_tensor.requires_grad = True
    target_class = args.target_class

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    image = np.array(image.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    Image.fromarray(visualization).save(f'{number2food[args.target_class]}_{model_name}.jpg')