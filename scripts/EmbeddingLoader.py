import torch
import pickle
from torch.nn import Embedding
from StyleClassifier import StyleClassifier

with open("style_classifier_params_300.pickle","rb") as f:
    dict_model=pickle.load(f)

embedding_weight=dict_model["embedding.weight"]

EmbeddingLayer=Embedding(embedding_weight.shape[0],embedding_weight.shape[1])
EmbeddingLayer.weight=torch.nn.Parameter(embedding_weight)

