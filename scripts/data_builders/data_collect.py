import numpy as np
import pandas as pd
import os

def sentence_truncature(sentence,wordCountLimit):
    list_words=sentence.split()
    center=len(list_words)//2
    l=wordCountLimit//2
    list_words=list_words[max(center-l,0):min(center+l+1,len(list_words))]
    return " ".join(list_words)

def yieldContext(data,index,style,contextMargin=2,wordCountLimit=50):
    index=int(index)
    irange=data[data.iloc[:,3]==data.iloc[index,3]].iloc[:,0] #Only sentences from the same play are considered
    imin=max(index-contextMargin,min(irange))
    imax=min(index+contextMargin,max(irange))
    if style=="original":
        style=1
    else:
        style=2
    list_context=list(data.iloc[imin:index,style])+list(data.iloc[index+1:(imax+1),style])
    str_context=" ".join(list_context)
    return sentence_truncature(str_context,wordCountLimit),min(wordCountLimit,len(" ".join(data.iloc[imin:index,style]).split()))

# print(yieldContext(data,0,"original"))

if __name__ == "__main__":

# ---------------------------------------------------------------------------- #
#                 Collecting relevant file names                               #
# ---------------------------------------------------------------------------- #

    modern = []
    original = []

    for x in os.listdir("../../data/raw_data"):
        if "modern" in x and "aligned" in x:
            modern.append(x)
        if "original" in x and "aligned" in x:
            original.append(x)

    original.sort()
    modern.sort()

# ---------------------------------------------------------------------------- #
#                 Scanning data files and storing data to csv                  #
# ---------------------------------------------------------------------------- #

    i = 0
    data_m = np.loadtxt("../../data/raw_data/"+modern[0],dtype="str",delimiter="\n")
    data_o = np.loadtxt("../../data/raw_data/"+original[0],dtype="str",delimiter="\n")
    data_c = np.ones(data_m.shape[0])*i
    i+=1


    for m,o in zip(modern[1:],original[1:]):
        text_m = np.loadtxt("../../data/raw_data/"+m,dtype="str",delimiter="\n")
        text_o = np.loadtxt("../../data/raw_data/"+o,dtype="str",delimiter="\n")
        data_m = np.concatenate((data_m,text_m),axis=0)
        data_o = np.concatenate((data_o,text_o),axis=0)
        data_c = np.concatenate((data_c,np.ones(text_m.shape[0])*i),axis=0)
        i+=1

    data = np.asarray([[x,y,z,c] for x,y,z,c in zip(range(data_m.shape[0]),data_m, data_o,data_c)])

    np.savetxt("../../data/shakespeare.csv",data,fmt="%s",delimiter="_")

    data=pd.read_csv("../../data/shakespeare.csv",sep="_",header=None)

# ---------------------------------------------------------------------------- #
#                           Prepare and save context                           #
# ---------------------------------------------------------------------------- #

    context={"index":[None for i in range(len(data))],
    "original":[None for i in range(len(data))],
    "modern":[None for i in range(len(data))],
    "play":[None for i in range(len(data))],
    "idx_original":[None for i in range(len(data))],
    "idx_modern":[None for i in range(len(data))],}

    for i in range(len(data)):
        context["index"][i] = data.iloc[i,0]
        context["original"][i],context["idx_original"][i] = yieldContext(data,i,"original")
        context["modern"][i],context["idx_modern"][i] = yieldContext(data,i,"modern")
        context["play"][i] = data.iloc[i,3]

    context=pd.DataFrame.from_dict(context)
    context.set_index("index")

    context.to_csv("../../data/context.csv",header=False,sep="_",index=False)
