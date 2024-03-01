# !/usr/bin/env python3
import torch

# Parse Input Data
with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Create vocubulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create encoder and decoder
# TLDR: every char gets an integer
# e.g. {' ': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, ...}
#      {0: ' ', 1: 'a', 2: 'b', 3: 'c', 4: 'd', ...}
#      "hi there" -> [46, 47, 1, 58, 46, 43, 56, 43]
#      [46, 47, 1, 58, 46, 43, 56, 43] -> "hi there"
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: "".join([itos[i] for i in x])

print("Encode 'hi there': ", encode("hi there"))
print(
    "Decode [46, 47, 1, 58, 46, 43, 56, 43]: ", decode([46, 47, 1, 58, 46, 43, 56, 43])
)

# Now lets store our data in a tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Now split our data into input and target
n = int(0.9 * len(data))  # 90% of the data
train_data = data[:n]
val_data = data[n:]
