import numpy as np
import os

# ---------------------------------------------------------------------------- #
#                 Collecting relevant file names                               #
# ---------------------------------------------------------------------------- #

modern = []
original = []

for x in os.listdir("../data"):
    if "modern" in x and "aligned" in x:
        modern.append(x)
    if "original" in x and "aligned" in x:
        original.append(x)

original.sort()
modern.sort()

# ---------------------------------------------------------------------------- #
#                 Scaning data files and storing data to csv                   #
# ---------------------------------------------------------------------------- #

i = 0
data_m = np.loadtxt("../data/"+modern[0],dtype="str",delimiter="\n")
data_o = np.loadtxt("../data/"+original[0],dtype="str",delimiter="\n")
data_c = np.ones(data_m.shape[0])*i
i+=1


for m,o in zip(modern[1:],original[1:]):
    text_m = np.loadtxt("../data/"+m,dtype="str",delimiter="\n")
    text_o = np.loadtxt("../data/"+o,dtype="str",delimiter="\n")
    data_m = np.concatenate((data_m,text_m),axis=0)
    data_o = np.concatenate((data_o,text_o),axis=0)
    data_c = np.concatenate((data_c,np.ones(text_m.shape[0])*i),axis=0)
    i+=1

data = np.asarray([[x,y,z,c] for x,y,z,c in zip(range(data_m.shape[0]),data_m, data_o,data_c)])

np.savetxt("../data/shakespeare.csv",data,fmt="%s",delimiter="_")
