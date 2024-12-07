from PIL import Image
import math
import json
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def resize_and_pad_image(image, target_resolution):
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image

def save_config(args, config_path):
    config = {
        'dataset': args.dataset,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'model_path': args.model_path
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def split_dataset(dataset, test_size=0.2, random_state=42):
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset