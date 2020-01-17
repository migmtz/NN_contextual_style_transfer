import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch

from collections import namedtuple

# ---------------------------------------------------------------------------- #
#                     Calculate the word frequency of plays                    #
# ---------------------------------------------------------------------------- #

def add_word(dict,word):
    if word not in dict:
        dict[word]=1
    else:
        dict[word]+=1
    return dict

def freq_update(freq,x):
    for word in x.split():
        word=word.upper()
        if word[-1] in ["!","?",".",",",";",":"]:
            freq=add_word(freq,word[:-1])
            freq=add_word(freq,word[-1])
        else:
            freq=add_word(freq,word)
    return freq

def select_most_frequent(d_res,dict,n):
    d = dict
    vals=list(d.values())
    keys=list(d.keys())
    v=max(vals)
    k=keys[vals.index(v)]
    d_res[k]=v
    if n==1:
        return d_res
    else:
        del d[k]
        return select_most_frequent(d_res,d,n-1)

# ---------------------------------------------------------------------------- #
#                     Create torch Dataset object to process data              #
# ---------------------------------------------------------------------------- #

def string2code(s,d):
    return torch.LongTensor([d[w] for w in s])

def code2string(t,d):
    if type(t) != list:
        t = t.tolist()
    return ' '.join(d[i] for i in t)

Batch = namedtuple("Batch", ["x", "y","ctx_x","ctx_y","len_x","len_y","len_ctx_x","len_ctx_y"])
class Shakespeare(Dataset):
    """
    Tensors returned when loaded in the dataloader:

    x_1 : modern english verse
    x_2 : shakespearian verse

    ctx_1 = context of the modern english verse
    ctx_2 = context of the shakespearian verse

    len_x : length of the modern english verse
    len_y : length of the shakespearian verse

    len_ctx_x : length of the modern english verse context
    len_ctx_y : length of the shakespearian verse context

    """

    def __init__(self, data,ctx,dict_words,device):
        i = 0
        self.device = device

        self.x = []
        self.y = []
        self.ctx_x = []
        self.ctx_y = []
        print("Loading ...")
        for sample,sample_ctx in zip(data,ctx):
            try:
                eng = string2code(sample[1].split(),dict_words).to(self.device)
                sha = string2code(sample[2].split(),dict_words).to(self.device)
                ctx_sha = string2code(sample_ctx[2].split(),dict_words).to(self.device)
                ctx_eng = string2code(sample_ctx[1].split(),dict_words).to(self.device)
                self.x.append(eng)
                self.y.append(sha)
                self.ctx_x.append(ctx_eng)
                self.ctx_y.append(ctx_sha)
            except:
                i+=1
        print("- Shakespeare dataset length : ",len(self.x))
        print("- Corrupted samples (ignored) : ",i)
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index],\
        self.ctx_x[index],self.ctx_y[index],\
        torch.LongTensor([self.x[index].shape[0]]).to(self.device),torch.LongTensor([self.y[index].shape[0]]).to(self.device),\
        torch.LongTensor([self.ctx_x[index].shape[0]]).to(self.device),torch.LongTensor([self.ctx_y[index].shape[0]]).to(self.device)

    @staticmethod
    def collate(batch):

        x = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True,padding_value=-1)
        y = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True,padding_value=-1)

        ctx_x = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True,padding_value=-1)
        ctx_y = torch.nn.utils.rnn.pad_sequence([item[3] for item in batch], batch_first=True,padding_value=-1)

        len_x = torch.cat([item[4] for item in batch])
        len_y = torch.cat([item[5] for item in batch])

        len_ctx_x = torch.cat([item[6] for item in batch])
        len_ctx_y = torch.cat([item[7] for item in batch])


        return Batch(x,y,ctx_x,ctx_y,len_x,len_y,len_ctx_x,len_ctx_y)

# ---------------------------------------------------------------------------- #
#                main function                                                 #
# ---------------------------------------------------------------------------- #

def prepare_dataset(device):
    """
    Input:
    a torch.device object

    Return :
    - a torch Dataset | class : Shakespeare inherited from torch.utils.data.Dataset
    - a python word dictionary (aka tokenizer) | class : dict

    Tensors returned when loaded in the dataloader:
    x_1 : modern english verse
    x_2 : shakespearian verse

    ctx_1 = context of the modern english verse
    ctx_2 = context of the shakespearian verse

    len_x : length of the modern english verse
    len_y : length of the shakespearian verse

    len_ctx_x : length of the modern english verse context
    len_ctx_y : length of the shakespearian verse context
    """


    #Load data
    data = np.loadtxt('data/shakespeare.csv',dtype="str",delimiter="_")
    ctx = np.loadtxt('data/context.csv',dtype="str",delimiter="_")

    #Plot word frequency
    tab=data[:,1:3].flatten()
    freq={}
    for x in tab:
        freq = freq_update(freq,x)
    #freq_top=select_most_frequent({},freq,50)
    #plt.hist(freq.values())

    #Create a word dictionnary
    dict_words={k:i for i,k in enumerate(freq.keys())}

    #preproessing data
    for sample,sample_ctx in zip(data,ctx):
        for sign in ["!","?",".",",",";",":","—"]:
            sample[1] = sample[1].replace(sign," "+sign+" ").upper()
            sample[2] = sample[2].replace(sign," "+sign+" ").upper()
            sample_ctx[1] = sample_ctx[1].replace(sign," "+sign+" ").upper()
            sample_ctx[2] = sample_ctx[2].replace(sign," "+sign+" ").upper()

    train_data = Shakespeare(data,ctx,dict_words,device)
    return train_data,dict_words
