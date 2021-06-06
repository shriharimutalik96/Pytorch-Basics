import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyper Paarmeters
input_size = 28
sequence_length = 28
hidden_size = 256
num_layers = 2
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Create d bi-directional LSTM
class BRNN(nn.Module):

    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(BRNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.brnn = nn.LSTM(input_size=input_size,
            hidden_size=hidden_size,num_layers=num_layers,
            batch_first=True,bidirectional=True)
        self.fc = nn.Linear(in_features=hidden_size*2,out_features=num_classes)


    def forward(self,x):
        h0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(device)
        x ,hidden_state = self.brnn(x,(h0,c0))
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)

        return x



# load data
train_dataset = datasets.MNIST(root="dataset/",train=True,
            transform=transforms.ToTensor(),download=True)

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)


test_dataset = datasets.MNIST(root="dataset/",train=False,
            transform=transforms.ToTensor(),download=True)

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)


model = BRNN(input_size,hidden_size,num_layers,num_classes).to(device)

# Set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# training loop
for epoch in range(num_epochs):

    for batch_idx,(data,targets) in enumerate(train_loader):
        data = data.to(device).squeeze(axis=1)
        targets = targets.to(device)


        scores = model(data)

        loss = criterion(scores,targets)


        optimizer.zero_grad()
        loss.backward()

        optimizer.step()



def check_accuracy(loader,model):

    num_correct  = 0
    num_samples = 0

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)

            _,predections = scores.max(1)
            num_correct += (predections==y).sum()
            num_samples = predections.size(0)



        print(f'Got {num_correct} / {num_samples} with accuracy {(float(num_correct)/float(num_samples))*100:.2f}')

    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)
