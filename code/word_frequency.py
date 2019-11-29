import ipdb
import os
from os import listdir

path="../data/shakespeare/enotes/merged"

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

def freq_update(freq,file):
    for line in file: 
        for word in line.split():
            word=word.upper()
            if word[-1] in ["!","?",".",",",";",":"]:
                freq=add_word(freq,word[:-1])
                freq=add_word(freq,word[-1])
            else:
                freq=add_word(freq,word)
    return freq


for f in listdir(path):
    print(f)
    with open(path+"/"+f, "r") as f:
        freq=freq_update(freq,f)
        f.close()


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
freq_top=select_most_frequent({},freq,50)

import matplotlib.pyplot as plt

plt.hist(freq.values())

# ---------------------------------------------------------------------------- #
#                            Create word dictionnary                           #
# ---------------------------------------------------------------------------- #

dict_words={}
i=0
for k in freq.keys():
    dict_words[k]=i
    i+=1
