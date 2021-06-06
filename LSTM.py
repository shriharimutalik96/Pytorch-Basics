# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper parameters
input_size = 28 # one row at a timestep
sequence_length=28 # number of time steps
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# Create a RNN
class LSTM(nn.Module):

    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm= nn.LSTM(input_size=input_size,hidden_size=hidden_size,
            num_layers=num_layers,batch_first=True)
        # N x time_sequences x features
        self.fc = nn.Linear(hidden_size*sequence_length,num_classes)
        # hidden_size*sequence_length : 256*28



    def forward(self,x):
        # Initialize the first hidden state
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        #print(f'H0 shape : {h0.shape}')
        # h0 shape : (2,64,256)

        # Forward prop
        out , hidden_state = self.lstm(x,(h0,c0))# ignore the hidden state
        out = out.reshape(out.shape[0],-1)
        #print(f'Out shape before fc : {out.shape}')
        out = self.fc(out)
        #print(f'Out shape after fc {out.shape}')

        return out

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

model = LSTM(input_size,hidden_size,num_layers,num_classes).to(device)

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# Train the network

for epoch in range(num_epochs):
    print('----------------------------------------------------')
    print('----------------------------------------------------')
    print('----------------------------------------------------')
    print('----------------------------------------------------')
    print('----------------------------------------------------')
    print('----------------------------------------------------')
    print('----------------------------------------------------')
    print('----------------------------------------------------')
    print('----------------------------------------------------')
    print(epoch)
    i = 0
    # get data to cuda if possible
    for batch_idx,(data,targets) in enumerate(train_loader):
        i += 1
        print(i)
        # print(data.shape)
        # torch.Size([64,1,28,28])
        data = data.to(device=device).squeeze(axis=1) # remove dimension from shape
        targets = targets.to(device=device)

        # print(data.shape)
        # torch.Size([64,28,28])

        # forward : executing the forward function
        scores = model(data)   # predicted output from the model : y_hat targets : correct targets
        loss = criterion(scores,targets)
        print(f'Scores shape : {scores.shape}')


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
            x = x.to(device=device).squeeze(axis=1)
            y = y.to(device=device)
            #x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {(float(num_correct)/float(num_samples))*100:.2f}')

    model.train()


check_accuracy(train_loader,model)
check_accuracy(test_loader,model)
