import pickle
from torch.nn import Embedding
with open("embedding_dic.pickle","rb") as f:
    weights=pickle.load(f)

import ipdb; ipdb.set_trace()
print(type(data))

em=Embedding(*data)