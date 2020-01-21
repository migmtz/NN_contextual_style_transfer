

import torch

# dataset

class CNN_classifier(torch.nn.Module):
    def __init__(self,d_embedding):
        super().__init__()
        self.conv_1 = torch.nn.Conv1d(d_embedding,3,kernel_size = 3, stride = 1)
        self.max_pool = torch.nn.MaxPool1d(3,2)
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(3,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.conv_1(x.transpose(1,2))
        x = self.max_pool(x)
        x = self.relu(x)
        x = torch.max(x,2)[0]
        x = self.linear(x)
        x = self.sigmoid(x)
        return(x)