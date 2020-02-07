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

from pathlib import Path
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

    def forward(self,x):
        return self.model(x)

class State:
    def __init__(self,model,optim):
        self.model = model
        self.optim = optim
        self.epoch = 0


if __name__=="__main__":

    #Loading data
    data_path = "../data/shakespeare.csv"
    train_data, dict_words = prepare_dataset(data_path,device,ratio=0.5,shuffle_ctx=False) #check with shift+tab to look at the data structure
    batch_size = 64
    dict_token = {b:a for a,b in dict_words.items()} #dict for code2string
    dict_size = len(dict_words)
    d_embedding = 768 #to plug in into BERT model

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True,collate_fn=train_data.collate)

    #Initializing model and optimizer
    savepath = Path("../data/models/style_classifier/v0")
    if savepath.is_file():
        state = torch.load(savepath)
    else:
        style_classifier = StyleClassifier(dict_size=dict_size,d_embedding=d_embedding,d_hidden=10).to(device)
        optimizer = optim.Adam(params=style_classifier.parameters(),lr=0.01)
        state = State(style_classifier,optimizer)

    epochs = 10
    loss_func=BCELoss()
    n = len(train_data.x) // batch_size

    #Training models
    writer = SummaryWriter("../data/runs/style_classifier_01")
    for epoch in range(state.epoch,epochs):
        total_loss,total_accuracy  = 0,0
        for x,_,_,_ , _,_ ,_,_, label,_ in train_loader:

            state.optim.zero_grad()
            x = state.model.forward(x)
            y = label.reshape(-1,1).float()
            loss = loss_func(x,y)
            loss.backward()
            state.optim.step()

            total_loss += loss.item()
            total_accuracy += 1 - ( (x>0.5).int() - y ).abs().sum().item() / batch_size

        #Vizualization
        print("Epoch \t",epoch+1," | Loss \t ",round(total_loss/n,2))
        writer.add_scalar('train_loss',total_loss/n,epoch+1)
        writer.add_scalar('train_accuracy',total_accuracy/n,epoch+1)

        #Saving model
        state.epoch += 1
        with savepath.open ( "wb" ) as fp:
            state.epoch = epoch + 1
            torch.save( state, fp )
