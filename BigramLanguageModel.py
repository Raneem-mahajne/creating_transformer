# import torch.nn as nn
# import torch
# from torch.nn import functional as F
# n_embd = 32
#
# class BigramLanguageModel(nn.Module):
#     def __init__(self, vocab_size):
#         super(BigramLanguageModel, self).__init__()
#         self.token_embedding = nn.Embedding(vocab_size, n_embd) #random initialization
#         self.position_embedding_table = nn.Embedding(block_size, n_embd)
#         self.lm_head = nn.Linear(n_embd, vocab_size)
#
#     def forward(self, xb, yb= None):
#
#         token_embedding = self.token_embedding(xb) # (B,T,C)
#         logits = self.lm_head(token_embedding) # (B, T, vocab_size)
#
#         if yb is None:
#             loss = None
#         else:
#
#             B, T, C = logits.shape
#             logits = logits.view(B*T, C)
#             targets = yb.view(B*T)
#             loss = F.cross_entropy(logits, targets) # cross entropy expects (N,C) returns N by documentation
#
#         return logits, loss
#
#
#     def generate(self, idx, max_new_tokens):
#         # idx is (B, T) array of indices in the current context
#         for _ in range(max_new_tokens):
#             # get the predictions
#             logits, loss = self(idx)
#             logits = logits[:, -1, :]  # becomes (B, C)
#
#             # apply softmax to get probabilities
#             probs = F.softmax(logits, dim=-1)  # (B, C)
#
#             # sample from the distribution
#             idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
#             # append sampled index to the running sequence
#             idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
#         return idx
