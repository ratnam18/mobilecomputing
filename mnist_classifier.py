import numpy as np
from tqdm import tqdm
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt


print("Loading datasets...")
MNIST_transform_invert = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomInvert(p=1)
])
MNIST_transform = transforms.Compose([
    transforms.ToTensor()
])
MNIST_trainval = datasets.MNIST('.', download=True, train=True, transform=MNIST_transform_invert)
MNIST_trainval_2 = datasets.MNIST('.', download=True, train=True, transform=MNIST_transform)
MNIST_train = Subset(MNIST_trainval, range(50000))
MNIST_train_2 = Subset(MNIST_trainval_2, range(50000))
MNIST_val = Subset(MNIST_trainval, range(50000,60000))
MNIST_val_2 = Subset(MNIST_trainval_2, range(50000,60000))
MNIST_train = torch.utils.data.ConcatDataset([MNIST_train_2, MNIST_train])
MNIST_val = torch.utils.data.ConcatDataset([MNIST_val_2, MNIST_val])
MNIST_test = datasets.MNIST('.', download=True, train=False, transform=MNIST_transform)
print("Completed!!")

#create model directory to save the trained model
if not os.path.exists('./model'):
    os.makedirs('./model')

#hyperparameters for training the model
batch_size = 32
learning_rate = 0.001
weight_decay = 1e-4
epoch = 40
model_path = './model/mnist_model.pth'

trainloader = DataLoader(MNIST_train, batch_size, shuffle=True)
valloader = DataLoader(MNIST_val, batch_size, shuffle=True)
testloader = DataLoader(MNIST_test, batch_size, shuffle=True)

#view sample data
examples = enumerate(testloader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)
plt.imshow(example_data[1][0], cmap ='gray')
plt.show()

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels= 120, kernel_size=(4,4), stride=(1,1), padding=(0,0))

        self.fully_connected1 = nn.Linear(in_features=120, out_features=84)
        self.fully_connected2=nn.Linear(in_features=84, out_features=10)

        self.pooling_layer = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):

        #Convolution Layer 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling_layer(x)
        
        #Convolution Layer 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling_layer(x)
        x = self.dropout(x)

        #Convolution Layer 3
        x = self.conv3(x)
        x = self.relu(x)

        #flatten x
        x = x.view(-1, 120)

        #Fully connected layer 1
        x = self.fully_connected1(x)
        x = self.relu(x)

        #Fully connected layer 2
        x = self.fully_connected2(x)

        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Network().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

training_loss = []
validation_loss = []

def train(model, loader, valloader, num_epoch = 10):
    print("Start training...")
    for i in tqdm(range(num_epoch)):
        model.train()
        train_running_loss = []
        for batch, label in loader:
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, label)
            train_running_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        print("Epoch {} training loss:{}".format(i+1,np.mean(train_running_loss)))
        training_loss.append(np.mean(train_running_loss))

        #run the model to evaluate on the validation dataset and get the validation accuracy
        model.eval()
        val_running_loss = []
        for val_batch, val_label in valloader:
            val_batch = val_batch.to(device)
            val_label = val_label.to(device)
            pred = model(val_batch)
            loss_val = criterion(pred, val_label)
            val_running_loss.append(loss_val.item())
        print("Epoch {} validation loss:{}".format(i+1,np.mean(val_running_loss)))
        validation_loss.append(np.mean(val_running_loss))
        print("Evaluation on training data...")
        evaluate(model, trainloader)
    #save the model once all the epochs are complete
    torch.save(model.state_dict(), model_path)
    print("Done!")

def plot_loss():
    x = np.arange(0, len(training_loss), 1, dtype=int)
    plt.plot(x, training_loss, label = 'Training Loss')
    plt.plot(x, validation_loss, label = 'Validation Loss')
    plt.title("Training and Validation Plots")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./loss_plot.png')
    plt.show()


def evaluate(model, loader):
    model.eval() 
    correct = 0
    with torch.no_grad():
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            correct += (torch.argmax(pred, dim=1) == label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc

train(model, trainloader, valloader, epoch)
plot_loss()
print("Evaluate on validation set...")
evaluate(model, valloader)
print("Evaluate on test set")
evaluate(model, testloader)
