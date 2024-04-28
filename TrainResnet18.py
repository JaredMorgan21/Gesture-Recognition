import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, Tensor
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.io import read_image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import datasets, transforms
from torchvision.transforms import v2
from MakeDatasetAnnotationsFile import *
import cv2

# Hyperparameters:
labels_file = 'labels_file'
batch_size = 10
learning_rate = 1e-3
epochs = 3


class CustomGestureDataset(Dataset):

    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image = read_image(img_path)
        image = image.repeat(3, 1, 1)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.float()
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)

        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.float()
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


## functions to show an image
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
    ])

    dataset = CustomGestureDataset(labels_file, transform=transform)

    generator1 = torch.Generator().manual_seed(42)

    train_subset, test_subset = random_split(dataset, [.9, .1], generator=generator1)
    train_subset, val_subset = random_split(train_subset, [.9, .1], generator=generator1)

    train_dataloader = DataLoader(train_subset, batch_size=batch_size)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size)

    ## get some random training images
    dataiter = iter(train_dataloader)
    images, labels = next(dataiter)

    ## show images
    imshow(torchvision.utils.make_grid(images))

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)

    for param in resnet18.parameters():
        param.requires_grad = False

    for param in resnet18.fc.parameters():
        param.requires_grad = True

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet18.parameters(), lr=learning_rate)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, resnet18, loss_fn, optimizer)
        test(test_dataloader, resnet18, loss_fn)
    print("Done!")

    torch.save(resnet18.state_dict(), "resnet18.pth")
    print("Saved PyTorch Model State to resnet18.pth")
