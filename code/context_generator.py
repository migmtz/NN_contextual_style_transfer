
# The context of a given sentence is defined as the neighbouring sentences (up to 2 neighbors) in the considered play. 

import pandas as pd
import numpy as np

data=pd.read_csv("./data/shakespeare.csv",sep="_",header=None)

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
    return sentence_truncature(str_context,wordCountLimit)

# print(yieldContext(data,0,"original"))

context={"index":[],"original":[],"modern":[],"play":[]}

for i in range(len(data)):
    context["index"].append(data.iloc[i,0])
    context["original"].append(yieldContext(data,i,"original"))
    context["modern"].append(yieldContext(data,i,"modern"))
    context["play"].append(data.iloc[i,3])

context=pd.DataFrame.from_dict(context)
context.set_index("index")

context.to_csv("./data/context.csv",header=False,sep="_",index=False)