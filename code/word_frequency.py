import pandas as pd
import numpy as np
import os
from os import listdir
import pickle

# ---------------------------------------------------------------------------- #
#                     Calculate the word frequency of plays                    #
# ---------------------------------------------------------------------------- #

freq={}

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



data=pd.read_csv("./data/shakespeare.csv",sep="_")
tab=np.array(data.iloc[:,1:3]).flatten()
for x in tab:
    freq=freq_update(freq,x)


print(len(freq))

# ---------------------------------------------------------------------------- #
#                           Create basis for encoding                          #
# ---------------------------------------------------------------------------- #


def select_most_frequent(d_res,dict,n):
    vals=list(dict.values())
    keys=list(dict.keys())
    v=max(vals)
    k=keys[vals.index(v)]
    d_res[k]=v
    if n==1:
        return d_res
    else:
        del dict[k]
        return select_most_frequent(d_res,dict,n-1)

# Most frequents words
#freq_top=select_most_frequent({},freq,50)

import matplotlib.pyplot as plt

plt.hist(freq.values())

# ---------------------------------------------------------------------------- #
#                Create  and save word dictionnary                             #
# ---------------------------------------------------------------------------- #

dict_words={}
i=0
for k in freq.keys():
    dict_words[k]=i
    i+=1

with open('./data/dict_words.pkl', 'wb') as f:
    pickle.dump(dict_words, f, pickle.HIGHEST_PROTOCOL)
