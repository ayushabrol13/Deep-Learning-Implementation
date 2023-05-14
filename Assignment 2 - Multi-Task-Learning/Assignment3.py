import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils, datasets, models
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(178, pad_if_needed=True),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_val_test = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


train_dataset = datasets.CelebA(
    root='data', split='train', transform=transform_train, download=True)
val_dataset = datasets.CelebA(
    root='data', split='valid', transform=transform_val_test, download=True)
test_dataset = datasets.CelebA(
    root='data', split='test', transform=transform_val_test, download=True)


train_indices, val_indices, test_indices = range(
    len(train_dataset)), range(len(val_dataset)), range(len(test_dataset))


labels = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair',
          'Brown_Hair', 'Male', 'Smiling', 'Young']
selected_indices = [i for i in range(len(labels))]


class SubsetCustom(Dataset):
    def __init__(self, dataset, indices, selected_indices):
        self.dataset = dataset
        self.indices = indices
        self.selected_indices = selected_indices

    def __getitem__(self, idx):
        image, attr = self.dataset[self.indices[idx]]
        return image, attr[self.selected_indices]

    def __len__(self):
        return len(self.indices)


train_dataset_new = SubsetCustom(
    train_dataset, train_indices, selected_indices)
val_dataset_new = SubsetCustom(val_dataset, val_indices, selected_indices)
test_dataset_new = SubsetCustom(test_dataset, test_indices, selected_indices)


train_loader = DataLoader(train_dataset_new, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset_new, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset_new, batch_size=128, shuffle=False)

model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, 8)
model.to(device)

optimizer_vgg = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
num_epochs = 5

train_loss_vgg = []
val_loss_vgg = []
train_acc_vgg = []
val_acc_vgg = []


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, train_loss, train_acc, val_loss_vgg, val_acc_vgg):
    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = 0
        train_acc_epoch = 0
        for (images, labels) in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.float()
            labels = labels.float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
            preds = torch.argmax(outputs, dim=1)
            labels = labels.argmax(dim=1, keepdim=True)
            train_acc_epoch += torch.sum(preds == labels).item()
        train_loss_epoch /= len(train_loader)
        train_acc_epoch /= len(train_loader.dataset)
        train_loss.append(train_loss_epoch)
        train_acc.append(train_acc_epoch)
        print("Epoch: {}/{}..".format(epoch+1, num_epochs),
                "Training Loss: {:.3f}..".format(train_loss_epoch),
                "Training Accuracy: {:.3f}".format(train_acc_epoch))
        
        model.eval()
        val_loss_epoch = 0
        val_acc_epoch = 0
        for (images, labels) in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.float()
            labels = labels.float()
            loss = criterion(outputs, labels)
            val_loss_epoch += loss.item()
            preds = torch.argmax(outputs, dim=1)
            labels = labels.argmax(dim=1, keepdims=True)
            val_acc_epoch += torch.sum(preds == labels).item()
        val_loss_epoch /= len(val_loader)
        val_acc_epoch /= len(val_loader.dataset)
        val_loss_vgg.append(val_loss_epoch)
        val_acc_vgg.append(val_acc_epoch)
        print("Epoch: {}/{}..".format(epoch+1, num_epochs),
                "Validation Loss: {:.3f}..".format(val_loss_epoch),
                "Validation Accuracy: {:.3f}".format(val_acc_epoch))


print("Training the model VGG16..")
train(model, train_loader, val_loader, optimizer_vgg, criterion,
      num_epochs, train_loss_vgg, train_acc_vgg, val_loss_vgg, val_acc_vgg)
print("Training complete!")

torch.save(model.state_dict(), 'models/model_vgg16.pth')
np.save('models/train_loss_vgg16.npy', np.array(train_loss_vgg))
np.save('models/val_loss_vgg16.npy', np.array(val_loss_vgg))
np.save('models/train_acc_vgg16.npy', np.array(train_acc_vgg))
np.save('models/val_acc_vgg16.npy', np.array(val_acc_vgg))

plt.figure(figsize=(10, 5))
plt.plot(train_loss_vgg, label='Train Loss')
plt.plot(val_loss_vgg, label='Val Loss')
plt.legend()
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('plots/loss_vgg16.png')

print("Testing the model VGG16..")
with torch.no_grad():
    model.eval()
    test_acc = 0
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        test_acc += torch.sum(preds == labels).item()
    test_acc /= len(test_loader.dataset)
    print("Test Accuracy: {:.3f}".format(test_acc))
