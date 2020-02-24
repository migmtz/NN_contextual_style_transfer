from scripts.data_builders.prepare_dataset import prepare_dataset,prepare_dataset_ctx,string2code,code2string

import torch
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import math
from torch.nn import BCELoss,CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device = ",device)

train_data, dict_words = prepare_dataset("data/shakespeare.csv",device,ratio=0.5,shuffle_ctx=False) #check with shift+tab to look at the data structure
batch_size = 64
dict_token = {b:a for a,b in dict_words.items()} #dict for code2string

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           shuffle=True,collate_fn=train_data.collate)
dict_size = len(dict_words)

print("- dict size : ",dict_size)

from transformers import AdamW,get_linear_schedule_with_warmup
epochs = 50
lr = 0.0001
n = len(train_data.x) // batch_size
num_warmup_steps = n*10
num_training_steps = (n*epochs*2)


#Load model, Embedding, and freeze Embedding
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'distilbert-base-uncased').to(device)
Embedding = torch.load("data/models/embeddings/v0").to(device)
model.train()

# BERT fine-tuning parameters
optimizer = AdamW(model.parameters(),lr=lr)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,num_training_steps=num_training_steps)  # PyTorch scheduleroptimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False

writer = SummaryWriter("runs/style_classifier_v1")
for epoch in range(epochs):
    total_loss,total_accuracy = 0,0
    i = 0
    for x,_,_,_,_,_,_,_,label,_ in train_loader:
        i+=1
        optimizer.zero_grad()

        mask = (x != dict_size).int()
        x = Embedding(x)
        loss,x = model.forward(inputs_embeds=x,
                          attention_mask = mask,
                          labels = label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        accuracy = 1 - (x.argmax(dim=1) - label).abs().sum().item() / x.shape[0]
        total_accuracy += accuracy

        print('| epoch {:3d} | {:5d}/{:5d} batches | '
              'loss {:5.2f} | total loss {:5.2f} | '
              'learning rate {:.0e} | total accuracy {:5.2f}'.format(
                epoch+1, i, n,loss.item(),total_loss/i,optimizer.param_groups[0]["lr"],total_accuracy/i))
    print('-' * 86)
    writer.add_scalar('train_loss',total_loss/n,epoch+1)
    writer.add_scalar('train_accuracy',total_accuracy/n,epoch+1)
    for tag, parm in model.named_parameters():
      if ("distilbert.transformer" in tag and "weight" in tag):
        print(tag)
        writer.add_histogram(tag[23:-7], parm.grad.data.cpu().numpy(), epoch)
      elif "weight" in tag and "classifier" in tag:
        writer.add_histogram(tag[:-7], parm.grad.data.cpu().numpy(), epoch)
    torch.save(model,"style_classifier_v1")
    torch.save(Embedding,"embedding_v1")
