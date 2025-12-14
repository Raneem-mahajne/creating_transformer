import torch.nn as nn
import torch
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size) #.Embedding converts token id to vectors

    def forward(self, xb, yb= None):
        logits = self.token_embedding(xb) # returns (B,T,C)
        if yb is None:
            loss = None
        else:

            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = yb.view(B*T)
            loss = F.cross_entropy(logits, targets) # cross entropy expects (N,C) returns N by documentation

        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
