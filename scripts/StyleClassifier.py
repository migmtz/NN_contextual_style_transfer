from data_builders.prepare_dataset import prepare_dataset,string2code,code2string,assemble


import torch
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import math
from torch.nn import BCELoss,CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pickle
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device = ",device)

class Transpose(torch.nn.Module):

    def __init__(self,dim0,dim1):
        super().__init__()
        self.dim0=dim0
        self.dim1=dim1

    def forward(self,x):
        return torch.transpose(x,dim0=self.dim0,dim1=self.dim1)

class Reduce(torch.nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim=dim

    def forward(self,x):
        x=torch.sum(x,dim=self.dim)
        return x.reshape(x.shape[0],x.shape[1])

class StyleClassifier(torch.nn.Module):
    def __init__(self,dict_size=18000,d_embedding=300,d_hidden=10):
        super(StyleClassifier,self).__init__()
        self.embedding=nn.Embedding(dict_size+1,d_embedding,padding_idx=dict_size)
        self.transpose=Transpose(1,2)
        self.conv1d=nn.Conv1d(d_embedding,d_hidden,3,1)
        self.reduce=Reduce(2)
        self.pool=nn.MaxPool1d(3,1)
        self.linear=nn.Linear(d_hidden,1)
        self.sigmoid=nn.Sigmoid()
        self.model=nn.Sequential(self.embedding,self.transpose,self.conv1d,self.pool,self.reduce,self.linear,self.sigmoid)
        
    def forward(self,x1,x2):
        return torch.cat((self.model(x1),self.model(x2)),dim=1)
        
if __name__=="__main__":
    PATH=os.getcwd()
    PATH="./embedding_params.pickle"
    train_data, dict_words = prepare_dataset(device,ratio=0.5,shuffle_ctx=True) #check with shift+tab to look at the data structure
    batch_size = 32
    dict_token = {b:a for a,b in dict_words.items()} #dict for code2string

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True,collate_fn=train_data.collate)
    dict_size = len(dict_words)
    d_embedding = 300 #cf. paper Y.Kim 2014 Convolutional Neural Networks for Sentence Classification
    import ipdb; ipdb.set_trace()
    print("- dict size : ",dict_size)

    epochs = 100
    style_classifier = StyleClassifier(dict_size=dict_size,d_embedding=300,d_hidden=10).to(device)
    optimizer = optim.Adam(params=style_classifier.parameters(),lr=0.01)
    loss_func=BCELoss()

    n = len(train_data.x) // batch_size

    for epoch in range(epochs):
        total_loss = 0
        i = 0
        for x,y,_,_ , _,_ ,_,_, label,_ in train_loader:
            i+=1
            optimizer.zero_grad()
            z=style_classifier.forward(x,y)
            label=label.reshape(-1,1)
            z_label=torch.cat((label,1-label),dim=1).float()
            loss=loss_func(z,z_label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('-' * 35)
        print('| epoch {:3d} | '
              'lr {:02.2f} | '
              'loss {:5.2f}'.format(
                epoch+1, optimizer.state_dict()["param_groups"][0]["lr"],
                round(total_loss,2)))
    with open("style_classifier_params.pickle","wb") as f:
        pickle.dump(style_classifier.state_dict(),f)
