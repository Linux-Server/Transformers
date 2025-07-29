import torch
from torch import nn
from torch.nn import functional as F


## Hyperparameter
batch_size = 4
block_size = 8 



## read the file : shakes.txt
with open("shakes.txt", "r") as f:
    text = f.read()
    
# get the vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)

#create mapping from char to int
stoi = {strs:i for i,strs in enumerate(chars)}
itos = {i:strs for i,strs in enumerate(chars)}


encode = lambda inp_txt : [stoi[i] for i in inp_txt]
decode = lambda inp_num : "".join([itos[i] for i in inp_num])


data = torch.tensor(encode(text))

## split the data
n = int(0.9* len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    return x , y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "eval"]:
        


## Biagram model
class BaigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim = 32)
        self.position_embeding = nn.Embedding(block_size, 32)
        self.lm_head = nn.Linear(in_features=32, out_features=vocab_size)
        
    def forward(self, idx, target=None):
        B,T = idx.shape
        token_embedding = self.token_embedding_table(idx) ## idx(B,T)(4,8) --> B,T,C (4,8,65)
        pos_emebedding = self.position_embeding(torch.arange(T))
        x = token_embedding + pos_emebedding
        logits = self.lm_head(x)

        if target is None:
            return logits   
        else:     
            B,T,C = logits.shape
            logits = logits.view(B*T, C) # (32,65)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
            return logits, loss
    
    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            logits= self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
            
    

m = BaigramModel(vocab_size=vocab_size)
logits , loss= m(xb,yb)
logits.shape , loss 

