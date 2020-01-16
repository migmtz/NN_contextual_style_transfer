# In this code you can find a general idea of the parameters, modules and networks as they are explained in the original paper.
# They haven't been tested using the dataset we have to this day (9 December)


# ---------------------------------------------------------------------------- #
#                        Parameters and Hyper parameters                       #
# ---------------------------------------------------------------------------- #

# --------------------------- Encoders and decoders -------------------------- #

nb_heads = 4
d_feedforward = 1024

batch_size = 64

# ---------------------------------------------------------------------------- #
#                                   Optimizer                                  #
# ---------------------------------------------------------------------------- #

from torch.optim import Adam

l_r = 5e-4
optimizer=Adam(lr=l_r)

# ---------------------------------------------------------------------------- #
#                                 Loss function                                #
# ---------------------------------------------------------------------------- #

# Weights of loss function
l1=1
l2=1
l3=1
l4=1
lambda_list = [l1,l2,l3,l4]

# PENDING !!!

# d_embedding

# dict_size

# ---------------------------------------------------------------------------- #
#                                    Layers                                    #
# ---------------------------------------------------------------------------- #

context_encoder = torch.nn.TransformerEncoderLayer(d_model = d_embedding, nhead = nb_heads, dim_feedforward = d_feedforward)

sentence_encoder = torch.nn.TransformerEncoderLayer(d_model = d_embedding, nhead = nb_heads, dim_feedforward = d_feedforward)



sentence_decoder = torch.nn.TransformerDecoderLayer(d_model = d_feedforward + 1, nhead = nb_heads, dim_feedforward = d_embedding)



linear_context = torch.nn.Sequential(
torch.nn.Linear(2*d_feed_forward,d_feedforward),
torch.nn.ReLU()
)


embed_layer = torch.nn.Embedding(dict_size,d_embedding)

# This is just to explicit that the optim is using Adam, the params are not to be kept like this
params = list(context_encoder.parameters()) + list(sentence_encoder.parameters()) + list(sentence_decoder.parameters()) + list(linear_context.parameters())
optim = torch.optim.Adam(params, lr = l_r)



# ---------------------------------------------------------------------------- #
#                                    Modules                                   #
# ---------------------------------------------------------------------------- #




# ----------------------------- Style classifier ----------------------------- #

# The Style Classifier uses a classical CNN network, this is just the one used in TME of AMAL, not yet modified or revised
# Feel free to modify / add your version

class Model(torch.nn.Module):
    def __init__(self,d_embedding):
        super().__init__()
        self.conv_1 = torch.nn.Conv1d(d_embedding,3,kernel_size = 3, stride = 1)
        self.max_pool = torch.nn.MaxPool1d(3,2)
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(3,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.conv_1(x.transpose(1,2))
        x = self.max_pool(x)
        x = self.relu(x)
        x = torch.max(x,2)[0]
        x = self.linear(x)
        x = self.sigmoid(x)
        return(x)
