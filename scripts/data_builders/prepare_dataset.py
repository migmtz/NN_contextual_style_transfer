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
    """
    - **Input**:
        - string : a sentence
        - dict : a tokenizer
    - **Return** :
        - a torch Longtensor (sentence tokenized)
    """
    return torch.LongTensor([d["<SOS>"]]+[d[w] for w in s]+[d["<EOS>"]])

def code2string(t,d):
    """
    - **Input**:
        - torch.Longtensor : a sentence tokenized
        - dict : a tokenizer
    - **Return** :
        - a string sentence
    """
    if type(t) != list:
        t = t.tolist()
    if len(d) in t:
        t = t[:t.index(max(t))]

    return ' '.join(d[i] for i in t)

# ---------------------------------------------------------------------------- #
#                1st dataset                                                   #
# ---------------------------------------------------------------------------- #
Batch = namedtuple("Batch", ["x", "y","ctx_x","ctx_y","len_x","len_y","len_ctx_x","len_ctx_y","label","label_ctx"])
class Shakespeare(Dataset):
    """
    Tensors returned when loaded in the dataloader:

    x_1 : input verse (modern / shakespearian)
    x_2 : output verse (modern / shakespearian)

    ctx_1 = context of the input verse
    ctx_2 = context of the output verse

    len_x : length of the input verse
    len_y : length of the output verse

    len_ctx_x : length of the input verse context
    len_ctx_y : length of the output verse context

    label : label of the input verse (0 : modern, 1 : shakespearian)
    label_ctx : label of the context (0 : wrong, 1 : right)

    """

    def __init__(self,data,dict_words,device,ratio=0.5,shuffle_ctx=False):
        i = 0
        self.device = device
        self.ratio = ratio
        self.shuffle_ctx = shuffle_ctx
        self.padding_value = len(dict_words)

        self.x = []
        self.y = []
        self.play = []

        print("Loading ...")
        for sample in data:
            try:
                eng = string2code(sample[1].split(),dict_words).to(self.device)
                sha = string2code(sample[2].split(),dict_words).to(self.device)
                self.x.append(eng)
                self.y.append(sha)
                self.play.append(sample[3].astype(float))
            except:
                print(sample[1])
                print(sample[2])
                i+=1
        print("- Shakespeare dataset length : ",len(self.x))
        print("- Corrupted samples (ignored) : ",i)
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: int):

        index_ctx,label_ctx = (np.random.randint(0,len(self.x)),0) if (self.shuffle_ctx and np.random.rand()<0.5) else (index,1)

        if (index_ctx == 0) or (self.play[index_ctx-1] != self.play[index_ctx]) :
            ctx_x = torch.cat([self.x[index_ctx + 1 ][1:-1],self.x[index_ctx + 2][1:]])
            ctx_y = torch.cat([self.y[index_ctx + 1 ][1:-1],self.y[index_ctx + 2][1:]])
        elif (index_ctx == len(self.x)-1) or (self.play[index_ctx+1] != self.play[index_ctx]) :
            ctx_x = torch.cat([self.x[index_ctx - 2][:-1],self.x[index_ctx - 1][1:-1]])
            ctx_y = torch.cat([self.y[index_ctx - 2][:-1],self.y[index_ctx - 1][1:-1]])
        else:
            ctx_x = torch.cat([self.x[index_ctx - 1][:-1],self.x[index_ctx + 1][1:]])
            ctx_y = torch.cat([self.y[index_ctx - 1][:-1],self.y[index_ctx + 1][1:]])


        if np.random.rand()<self.ratio:
            return self.x[index], self.y[index],\
            ctx_x,ctx_y,\
            torch.LongTensor([self.x[index].shape[0]]).to(self.device),torch.LongTensor([self.y[index].shape[0]]).to(self.device),\
            torch.LongTensor([ctx_x.shape[0]]).to(self.device),torch.LongTensor([ctx_y.shape[0]]).to(self.device),\
            torch.LongTensor([0]).to(self.device),torch.LongTensor([label_ctx]).to(self.device)
        else:
            return self.y[index], self.x[index],\
            ctx_y,ctx_x,\
            torch.LongTensor([self.y[index].shape[0]]).to(self.device),torch.LongTensor([self.x[index].shape[0]]).to(self.device),\
            torch.LongTensor([ctx_y.shape[0]]).to(self.device),torch.LongTensor([ctx_x.shape[0]]).to(self.device),\
            torch.LongTensor([1]).to(self.device),torch.LongTensor([label_ctx]).to(self.device)

    #@staticmethod
    def collate(self,batch):
        x = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True,padding_value=self.padding_value)
        y = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True,padding_value=self.padding_value)

        len_x = torch.cat([item[4] for item in batch])
        len_y = torch.cat([item[5] for item in batch])

        ctx_x = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True,padding_value=self.padding_value)
        ctx_y = torch.nn.utils.rnn.pad_sequence([item[3] for item in batch], batch_first=True,padding_value=self.padding_value)

        len_ctx_x = torch.cat([item[6] for item in batch])
        len_ctx_y = torch.cat([item[7] for item in batch])

        label = torch.cat([item[8] for item in batch])
        label_ctx = torch.cat([item[9] for item in batch])

        return Batch(x,y,ctx_x,ctx_y,len_x,len_y,len_ctx_x,len_ctx_y,label,label_ctx)

def prepare_dataset(path,device,ratio=0.5,shuffle_ctx=False):
    """
    - **Input**:
        - path : relative path of the shakespeare.csv file
        - device : a torch.device object
        - ratio : a float ratio between 0 and 1 that determines the average proportion of modern english verses in the data loader
        - shuffle_ctx : if `True`, shuffle the contexts within a Batch so that half of the inputs has a wrong context. Useful to train the context recognizer model.
    - **Return** :
        - a torch Dataset | class : Shakespeare inherited from torch.utils.data.Dataset
        - a python word dictionary (aka tokenizer) | class : dict
    - **Tensors returned when loaded in the dataloader**:
        - x_1 : input verse (modern / shakespearian)
        - x_2 : output verse (modern / shakespearian)

        - ctx_1 = context of the input verse
        - ctx_2 = context of the output verse

        - len_x : length of the input verse
        - len_y : length of the output verse

        - len_ctx_x : length of the input verse context
        - len_ctx_y : length of the output verse context

        - label : label of the input verse (0 : modern, 1 : shakespearian)
        - label_ctx : label of the context (0 : wrong, 1 : right)
    """


    #Load data
    data = np.loadtxt(path,dtype="str",delimiter="_")

    #Create a word dictionnary
    dict_words={}
    dict_words["<SOS>"] = 0
    dict_words["<EOS>"] = 1

    #preproessing data
    for sample in data:
        for sign in ["!","?",".",",",";"]:
            sample[1] = sample[1].replace(sign," "+sign+" ").upper()
            sample[2] = sample[2].replace(sign," "+sign+" ").upper()
        for sign in [":","—","“","”"]:
            sample[1] = sample[1].replace(sign,"").upper()
            sample[2] = sample[2].replace(sign,"").upper()

        for word in sample[1].split():
            if word not in dict_words:
                dict_words[word] = len(dict_words)
        for word in sample[2].split():
            if word not in dict_words:
                dict_words[word] = len(dict_words)

    train_data = Shakespeare(data,dict_words,device,ratio,shuffle_ctx)
    return train_data,dict_words

# ---------------------------------------------------------------------------- #
#                2nd dataset (for coherence)                                   #
# ---------------------------------------------------------------------------- #

Batch_ctx = namedtuple("Batch_ctx", ["ctx","pos_token","pos_ctx","label_ctx"])
class Shakespeare_ctx(Dataset):
    """
    Tensors returned when loaded in the dataloader:

    ctx = context with input sentence concatenated inside

    pos_token : position of each word in ctx

    pos_ctx : whether if a word belongs to the context or the input sentence

    label_ctx : label of the context (0 : wrong, 1 : right)

    """

    def __init__(self,data,dict_words,device,ratio=0.5,shuffle_ctx=False):
        i = 0
        self.device = device
        self.ratio = ratio
        self.shuffle_ctx = shuffle_ctx
        self.padding_value = len(dict_words)

        self.x = []
        self.y = []
        self.play = []

        print("Loading ...")
        for sample in data:
            try:
                eng = string2code(sample[1].split(),dict_words).to(self.device)
                sha = string2code(sample[2].split(),dict_words).to(self.device)
                self.x.append(eng)
                self.y.append(sha)
                self.play.append(sample[3].astype(float))
            except:
                print(sample[1])
                print(sample[2])
                i+=1
        print("- Shakespeare context dataset length : ",len(self.x))
        print("- Corrupted samples (ignored) : ",i)
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: int):

        index_ctx,label_ctx = (np.random.randint(0,len(self.x)),0) if (self.shuffle_ctx and np.random.rand()<0.5) else (index,1)
        x = self.x if np.random.rand()<self.ratio else self.y

        if (index_ctx == 0) or (self.play[index_ctx-1] != self.play[index_ctx]) :
            ctx = torch.cat([x[index][:-1],x[index_ctx + 1 ][1:-1],x[index_ctx + 2][1:]])
            pos_ctx = torch.LongTensor([1]*x[index][:-1].shape[0]+[0]*(x[index_ctx + 1 ][1:-1].shape[0]+x[index_ctx + 2][1:].shape[0]))
        elif (index_ctx == len(self.x)-1) or (self.play[index_ctx+1] != self.play[index_ctx]) :
            ctx = torch.cat([x[index_ctx - 2][:-1],x[index_ctx - 1][1:-1],x[index][1:]])
            pos_ctx = torch.LongTensor([0]*(x[index_ctx - 2][:-1].shape[0]+x[index_ctx - 1][1:-1].shape[0])+[1]*x[index][1:].shape[0])
        else:
            ctx = torch.cat([x[index_ctx - 1][:-1],x[index][1:-1],x[index_ctx + 1][1:]])
            pos_ctx = torch.LongTensor([0]*x[index_ctx - 1][:-1].shape[0]+[1]*x[index][1:-1].shape[0]+[0]*x[index_ctx + 1][1:].shape[0])

        pos_token = torch.LongTensor([i for i in range(ctx.shape[0])])
        label = torch.LongTensor([label_ctx])

        return ctx.to(self.device),\
        pos_token.to(self.device),\
        pos_ctx.to(self.device),\
        label.to(self.device)

    def collate(self,batch):
        ctx = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True,padding_value=self.padding_value)
        pos_token = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True,padding_value=0)
        pos_ctx = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True,padding_value=0)
        label = torch.cat([item[3] for item in batch])


        return Batch_ctx(ctx,pos_token,pos_ctx,label)

def prepare_dataset_ctx(path,device,ratio=0.5,shuffle_ctx=False):
    """
    - **Input**:
        - device : a torch.device object
        - ratio : a float ratio between 0 and 1 that determines the average proportion of modern english verses in the data loader
        - shuffle_ctx : if `True`, shuffle the contexts within a Batch so that half of the inputs has a wrong context. Useful to train the context recognizer model.
    - **Return** :
        - a torch Dataset | class : Shakespeare_ctx inherited from torch.utils.data.Dataset
        - a python word dictionary (aka tokenizer) | class : dict
    - Tensors returned when loaded in the dataloader:
        - ctx = context with input sentence concatenated inside
        - pos_token : position of each word in ctx
        - pos_ctx : whether if a word belongs to the context or the input sentence
        - label_ctx : label of the context (0 : wrong, 1 : right)
    """

    #Load data
    data = np.loadtxt(path,dtype="str",delimiter="_")

    #Create a word dictionnary
    dict_words={}
    dict_words["<SOS>"] = 0
    dict_words["<EOS>"] = 1

    #preproessing data
    for sample in data:
        for sign in ["!","?",".",",",";"]:
            sample[1] = sample[1].replace(sign," "+sign+" ").lower()
            sample[2] = sample[2].replace(sign," "+sign+" ").lower()
        for sign in [":","—","“","”"]:
            sample[1] = sample[1].replace(sign,"").lower()
            sample[2] = sample[2].replace(sign,"").lower()

        for word in sample[1].split():
            if word not in dict_words:
                dict_words[word] = len(dict_words)
        for word in sample[2].split():
            if word not in dict_words:
                dict_words[word] = len(dict_words)

    train_data = Shakespeare_ctx(data,dict_words,device,ratio,shuffle_ctx)
    return train_data,dict_words
