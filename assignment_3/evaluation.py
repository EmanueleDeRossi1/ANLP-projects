import torch

from utils import char_tensor
import string
import numpy as np
from model.model import LSTM

from language_model import generate


def compute_bpc(model, string):
    """
    Given a model and a string of characters, compute bits per character
    (BPC) using that model.

    Args:
        model: RNN-based model (RNN, LSTM, GRU, etc.)
        string: string of characters

    Returns:
        BPC for that set of string.
    """
    ################# STUDENT SOLUTION ################################
    # YOUR CODE HERE
    bpc_sum = 0
    
    with torch.no_grad():
        for i in range(len(string) - 1):
            
            input_string = string[i]
            target = string[i + 1]
            tensor_string = char_tensor(input_string)
            (hidden, cell) = decoder.init_hidden()
            
            output, (hidden, cell) = decoder.forward(tensor_string, (hidden, cell))
            
            softmax_output = torch.nn.functional.softmax(output, dim=1)
            target_index = all_characters.index(target)
            char_target_softmax = softmax_output[0][target_index].item()
            
            bpc_sum += np.log2(char_target_softmax)
            
        bpc = - bpc_sum / len(string)
            
        return bpc
    ###################################################################


all_characters = string.printable
n_characters = len(all_characters)

n_epochs = 3000
print_every = 100
plot_every = 10
hidden_size = 128
n_layers = 2

lr = 0.005
decoder = LSTM(n_characters, hidden_size, n_characters, n_layers)

string = "Here I am"

print(compute_bpc(decoder, string))