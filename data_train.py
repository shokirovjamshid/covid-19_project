import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from torchvision.models import resnet50

def path(folder_path):
    dir = Path(folder_path) # file_path  C:\Users\Shacsim_Systems\Desktop\project_python\project\COVID-19_project\dataset_covid19
    dir_train = dir/'train'
    dir_test = dir/'val'
    train_list = list(dir_train.glob('*/*.*'))
    test_list = list(dir_test.glob('*/*.*'))
    return train_list,test_list

def transforms():
    transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Update mean and std for 3 channels
    ])
    return transform

class COVID19Dataset(Dataset):
  def __init__(self,path,transform = None):
    self.data_list = list(path)
    self.transform = transform

  def __getitem__(self,idx):
    img = Image.open(self.data_list[idx])
    img = img.convert('RGB')
    if self.transform:
          img = self.transform(img)
    if 'Not affected by COVID-19' == str(self.data_list[idx]).split('\\')[-2]:
      label = 1
    elif 'Affected by COVID-19' == str(self.data_list[idx]).split('\\')[-2]:
      label = 0

    return img,torch.tensor(label)
  def __len__(self):
    return len(self.data_list)

def datalaoder(train_data,test_data):
    train_data_loader = DataLoader(train_data,
                        batch_size=19,
                        shuffle=True)
    test_data_loader = DataLoader(test_data,
                        batch_size=19,
                        shuffle=True)
    return train_data_loader,test_data_loader

def mlco():
    model = resnet50(num_classes=2)
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    return model,learning_rate,criterion,optimizer

train_on_gpu = torch.cuda.is_available()
device = torch.device("cpu")
if not train_on_gpu:
    print('CUDA is not available. Training on CPU...')
else:
    print('CUDA is available. Training on GPU')

if train_on_gpu:
    device = torch.device("cuda")

class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, criterion, optimizer, epochs=3):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer

        self.epochs = epochs
        self.history = {'loss':[],'acc':[],'val_loss':[],'val_acc':[]}

    def train_loop(self):
        sum_loss = 0
        sum_accuracy = 0
        n = len(self.train_dataloader)
        for i, (data,label) in enumerate(tqdm(self.train_dataloader)):
            data = data.to(device)
            label = label.to(device)
            # prediction model
            output = self.model(data)
            # find loss
            loss = self.criterion(output, label)

            sum_loss += loss.item()
            n_corrects = (output.argmax(axis=1)==label).sum().item()
            sum_accuracy += n_corrects/label.size(0)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


        train_loss = sum_loss/n
        train_accuracy = sum_accuracy/n

        self.history['loss'].append(train_loss)
        self.history['acc'].append(train_accuracy)

        return train_loss, train_accuracy

    def validation_loop(self):
        sum_loss = 0
        sum_accuracy = 0
        n = len(self.test_dataloader)
        for i, (data,label) in enumerate(tqdm(self.test_dataloader)):
            data = data.to(device)
            label = label.to(device)
            # prediction model
            output = self.model(data)
            # find loss
            loss = self.criterion(output, label)
            n_corrects = (output.argmax(axis=1)==label).sum().item()

            sum_loss += loss.item()
            sum_accuracy += n_corrects/label.size(0)

        val_loss = sum_loss/n
        val_accuracy = sum_accuracy/n

        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_accuracy)

        return val_loss, val_accuracy

    def train(self):
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_loop()
            val_loss, val_acc = self.validation_loop()
            print()
            print(f'Epoch[{epoch+1}/{self.epochs}] \t train_loss: {train_loss:.5f}, train_acc: {train_acc:.2f} \t val_loss: {val_loss:.5f} \t val_acc: {val_acc:.2}')

train_list,test_list = path("C:/Users/Shacsim_Systems/Desktop/project_python/project/COVID-19_project/dataset_covid19")
# print(train_list[1:9])
transform = transforms()
train_data = COVID19Dataset(train_list, transform)
test_data = COVID19Dataset(test_list, transform)
print(train_data[0])
train_data_loader,test_data_loader = datalaoder(train_data,test_data)
model,learning_rate,criterion,optimizer = mlco()

epochs=30
trainer = Trainer(
    model = model.to(device),
    train_dataloader=train_data_loader,
    test_dataloader=test_data_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=epochs,
)

trainer.train()