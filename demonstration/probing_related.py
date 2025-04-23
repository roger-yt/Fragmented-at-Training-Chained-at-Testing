from transformers import AutoTokenizer

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

class probing_MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(probing_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(1000, output_size)

    def forward(self, x):
        # print("x=", x)
        # print("y=", y)
        x = self.fc1(x)
        # loss_fct = CrossEntropyLoss()
        # loss = loss_fct(x, y)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x

Tokenizer = AutoTokenizer.from_pretrained("gpt2")
Tokenizer.pad_token = Tokenizer.eos_token
# def pre_val_task(letters, vals, test_pos):
#         return vals[test_pos-1][-1]-90

def my_task(letters, vals, pos):
        return Tokenizer.encode(str(vals[pos][-1]))
# def this_val_task(letters, vals, test_pos):
#         return Tokenizer.encode(str(vals[test_pos][-1]))

# def pre_val_task(letters, vals, test_pos):
#         return Tokenizer.encode(str(vals[test_pos-1][-1]))

# def prepre_val_task(letters, vals, test_pos):
#         return Tokenizer.encode(str(vals[test_pos-3][-1]))