import pandas as pd
import torch
import torchvision
from torch import nn, Tensor
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.io import read_image
from torchvision.models import resnet152, ResNet152_Weights
from torchvision import datasets, transforms
from torchvision.transforms import v2
from MakeDatasetAnnotationsFile import *

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
        image = image.repeat(3,1,1)
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
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

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

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    resnet152 = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)

    for param in resnet152.parameters():
        param.requires_grad = False

    for param in resnet152.fc.parameters():
        param.requires_grad = True

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet152.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, resnet152, loss_fn, optimizer)
        test(test_dataloader, resnet152, loss_fn)
    print("Done!")

    torch.save(resnet152.state_dict(), "resnet_152.pth")
    print("Saved PyTorch Model State to resnet_152.pth")