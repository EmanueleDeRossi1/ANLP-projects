import torch
import torch.nn as nn


# Here is a pseudocode to help with your LSTM implementation. 
# You can add new methods and/or change the signature (i.e., the input parameters) of the methods.
class LSTM(nn.Module):
    def __init__(self, n_characters, hidden_size, output_size, n_layers=1):
        """Think about which (hyper-)parameters your model needs; i.e., parameters that determine the
        exact shape (as opposed to the architecture) of the model. There's an embedding layer, which needs 
        to know how many elements it needs to embed, and into vectors of what size. There's a recurrent layer,
        which needs to know the size of its input (coming from the embedding layer). PyTorch also makes
        it easy to create a stack of such layers in one command; the size of the stack can be given
        here. Finally, the output of the recurrent layer(s) needs to be projected again into a vector
        of a specified size."""
        ############################ STUDENT SOLUTION ############################
        super(LSTM, self).__init__()
        self.n_characters = n_characters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Define layers
        self.embedding = nn.Embedding(n_characters, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        ##########################################################################

    def forward(self):
        """Your implementation should accept input character, hidden and cell state,
        and output the next character distribution and the updated hidden and cell state."""
        ############################ STUDENT SOLUTION ############################
        # YOUR CODE HERE
        # initialite embedding layer
        # input.view() reshapes the input tensor to fit into the embedding layer's expected shape. 1 is batch dimension, -1 means the size is inferred based on input size
        embedded = self.embedding(input.view(1, -1))  # Embedding layer
        output, hidden_cell = self.lstm(embedded.view(1, 1, -1), hidden_cell)  # LSTM layer
        output = self.fc(output.view(1, -1))  # Fully connected layer
        
        return output, hidden_cell

        ##########################################################################
        pass

    def init_hidden(self):
        """Finally, you need to initialize the (actual) parameters of the model (the weight
        tensors) with the correct shapes."""
        ############################ STUDENT SOLUTION ############################
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, 1, self.hidden_size).zero_(),
              weight.new(self.n_layers, 1, self.hidden_size).zero_())
        return hidden
        ##########################################################################
