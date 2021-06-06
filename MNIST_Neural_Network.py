# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create Fully Connected Neural Network
class NN(nn.Module):

    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()

        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,num_classes)


    def forward(self,x):
        #print(f'Input shape : {x.shape}')
        x = F.relu(self.fc1(x))
        #print(f'Hidden Layer shape : {x.shape}')
        x = self.fc2(x)
        #print(f'OutPut shape : {x.shape}')
        return x

#Set Device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Load Data

train_dataset = datasets.MNIST(root='dataset/',
                train=True,transform=transforms.ToTensor(),
                download=True)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_dataset = datasets.MNIST(root='dataset',
                train=False,transform=transforms.ToTensor(),
                download=True)

test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

# Initialize the network

model = NN(input_size=input_size,num_classes=num_classes).to(device)

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# Train the network

for epoch in range(num_epochs):
    print('----------------------------------------------------')
    print('----------------------------------------------------')
    print('----------------------------------------------------')
    print(epoch)
    # get data to cuda if possible
    for batch_idx,(data,targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # get to correct shape
        data = data.reshape(data.shape[0],-1)
        #print(data.shape)

        # forward
        scores = model(data)   # predicted output from the model : y_hat targets : correct targets
        loss = criterion(scores,targets)


        # backward
        optimizer.zero_grad()   # we are setting all the gradients to zero from the previous epoch
        loss.backward()

        # gradient descent or adam's step
        optimizer.step()  # updating the weights


# Check accuracy on training and test to see how good our model is

def check_accuracy(loader,model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x , y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {(float(num_correct)/float(num_samples))*100:.2f}')

    model.train()


check_accuracy(train_loader,model)
check_accuracy(test_loader,model)
