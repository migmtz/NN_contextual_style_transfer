from data_builders.prepare_dataset import prepare_dataset,prepare_dataset_ctx,string2code,code2string,assemble
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device = ",device)

#Preprocessing data
train_data, dict_words = prepare_dataset_ctx("data/shakespeare.csv",device,ratio=0.5,shuffle_ctx=True) #check with shift+tab to look at the data structure
batch_size = 64
dict_token = {b:a for a,b in dict_words.items()} #dict for code2string
dict_size = len(dict_words)
print("- dict size : ",dict_size)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           shuffle=True,collate_fn=train_data.collate)
