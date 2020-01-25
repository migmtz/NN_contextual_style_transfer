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
def assemble(sentence,context,function):
    """
    - **Input**:
        - torch.Longtensor: a sentence
        - torch.Longtensor: the sentence context
    - **Return**:
        - torch.Longtensor: the sentence within its context
    """
    #Questions:
    # faut il concaténer tel quels?
    # les insérer dans le contexte (suggéré par l'article)?
    #  Et en ce cas, faut il supprimer le padding sur la phrase (voir le supprimer altogether

    #Calcul de l'index auquel insérer le deuxième élément
    # index=(context in string2code(".!?—")).nonzero()[1]

    return torch.cat((function(sentence),context),dim=1)


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
    t = t[1:t.index(min(t))-1]

    return ' '.join(d[i] for i in t)

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

    """

    def __init__(self, data,ctx,dict_words,device,ratio=0.5,shuffle_ctx=False):
        i = 0
        self.device = device
        self.ratio = ratio
        self.shuffle_ctx = shuffle_ctx
        self.padding_value = len(dict_words°+1

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
        index_ctx,label_ctx = (np.random.randint(0,len(self.x)),0) if (self.shuffle_ctx and np.random.rand()<0.5) else (index,1)


        if np.random.rand()<self.ratio:
            return self.x[index], self.y[index],\
            self.ctx_x[index_ctx],self.ctx_y[index],\
            torch.LongTensor([self.x[index].shape[0]]).to(self.device),torch.LongTensor([self.y[index].shape[0]]).to(self.device),\
            torch.LongTensor([self.ctx_x[index_ctx].shape[0]]).to(self.device),torch.LongTensor([self.ctx_y[index].shape[0]]).to(self.device),\
            torch.LongTensor([0]).to(self.device),torch.LongTensor([label_ctx]).to(self.device)
        else:
            return self.y[index], self.x[index],\
            self.ctx_y[index_ctx],self.ctx_x[index],\
            torch.LongTensor([self.y[index].shape[0]]).to(self.device),torch.LongTensor([self.x[index].shape[0]]).to(self.device),\
            torch.LongTensor([self.ctx_y[index_ctx].shape[0]]).to(self.device),torch.LongTensor([self.ctx_x[index].shape[0]]).to(self.device),\
            torch.LongTensor([1]).to(self.device),torch.LongTensor([label_ctx]).to(self.device)

    @staticmethod
    def collate(batch):

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

# ---------------------------------------------------------------------------- #
#                main function                                                 #
# ---------------------------------------------------------------------------- #

def prepare_dataset(device,ratio=0.5,shuffle_ctx=False):
    """
    - **Input**:
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
    dict_words={k:i+2 for i,k in enumerate(freq.keys())}
    dict_words["<SOS>"] = 0
    dict_words["<EOS>"] = 1

    #preproessing data
    for sample,sample_ctx in zip(data,ctx):
        for sign in ["!","?",".",",",";",":","—"]:
            sample[1] = sample[1].replace(sign," "+sign+" ").upper()
            sample[2] = sample[2].replace(sign," "+sign+" ").upper()
            sample_ctx[1] = sample_ctx[1].replace(sign," "+sign+" ").upper()
            sample_ctx[2] = sample_ctx[2].replace(sign," "+sign+" ").upper()

    train_data = Shakespeare(data,ctx,dict_words,device,ratio,shuffle_ctx)
    return train_data,dict_words
