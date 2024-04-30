import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision.models import resnet152, resnet50, resnet18, vit_b_16
from torchvision.transforms import v2
from torchvision import datasets, transforms

from MakeDatasetAnnotationsFile import idxs_to_class_names
from TrainResnet152 import CustomGestureDataset, test

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

model_choices = [
    'resnet_152.pth',
    'resnet50.pth',
    'resnet18.pth',
    'vit_b_16.pth'
]

labels_file = 'labels_file'
transform = transforms.Compose([
    transforms.Resize([224, 224]),
])
dataset = CustomGestureDataset(labels_file, transform=transform)

generator1 = torch.Generator().manual_seed(42)

train_subset, test_subset = random_split(dataset, [.9, .1], generator=generator1)
train_subset, val_subset = random_split(train_subset, [.9, .1], generator=generator1)

batch_size = 10 # same as during training
train_dataloader = DataLoader(train_subset, batch_size=batch_size)
val_dataloader = DataLoader(val_subset, batch_size=batch_size)
test_dataloader = DataLoader(test_subset, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()

for model_name in model_choices:

    print("Evaluating: " + model_name)

    if model_name == 'resnet_152.pth':
        resnet152 = resnet152().to(device)
        resnet152.load_state_dict(torch.load('resnet_152.pth'))
        test(test_dataloader, resnet152, loss_fn)
    elif model_name == 'resnet50.pth':
        resnet50 = resnet50().to(device)
        resnet50.load_state_dict(torch.load('resnet50.pth'))
        test(test_dataloader, resnet50, loss_fn)
    elif model_name == 'resnet18.pth':
        resnet18 = resnet18().to(device)
        resnet18.load_state_dict(torch.load('resnet18.pth'))
        test(test_dataloader, resnet18, loss_fn)
    elif model_name == 'vit_b_16.pth':
        transforms = v2.Compose([
            v2.Resize((224, 224))
        ])

        dataset = CustomGestureDataset(labels_file, transform=transforms)

        generator1 = torch.Generator().manual_seed(42)

        train_subset, test_subset = random_split(dataset, [.9, .1], generator=generator1)
        test_dataloader = DataLoader(test_subset, batch_size=batch_size)

        vit_b_16 = vit_b_16().to(device)
        vit_b_16.load_state_dict(torch.load('vit_b_16.pth'))
        test(test_dataloader, vit_b_16, loss_fn)