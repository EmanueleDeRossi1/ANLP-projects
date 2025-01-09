import numpy as np
import numpy.typing as npt

from model.model_utils import softmax, relu, relu_prime
from typing import Tuple



class NeuralNetwork(object):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int, 
        num_classes: int,
        seed: int = 1
    ):
        """
        Initialize neural network's weights and biases.
        """
        ############################# STUDENT SOLUTION ####################
        # YOUR CODE HERE
        #     TODO:
        #         1) Set a seed so that your model is reproducible
        #         2) Initialize weight matrices and biases with uniform
        #         distribution in the range (-1, 1).
        np.random.seed(seed)
        self.weights_1 = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.bias_1 = np.random.uniform(-1, 1, (hidden_size, 1))
        
        self.weights_2 = np.random.uniform(-1, 1, (num_classes, hidden_size))
        self.bias_2 = np.random.uniform(-1, 1, (num_classes, 1))
        
        ###################################################################

    def forward(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Forward pass with X as input matrix, returning the model prediction
        Y_hat.
        """
        ######################### STUDENT SOLUTION #########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Perform only a forward pass with X as input.
        
        z1 = np.dot(self.weights_1, X) + self.bias_1
        a1 = relu(z1)
        z2 = np.dot(self.weights_2, a1) + self.bias_2
        a2 = softmax(z2)

        return a2
        

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Create a prediction matrix with `self.forward()`
        """
        ######################### STUDENT SOLUTION ###########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Create a prediction matrix of the intent data using
        #         `self.forward()` function. The shape of prediction matrix
        #         should be similar to label matrix produced with
        #         `labels_matrix()`
        probabilities = self.forward(X)
        
        # predictions = probabilities.argmax(axis=0)
        
        max_indices = np.argmax(probabilities, axis=0)
        predictions = np.zeros_like(probabilities)
        
        predictions[max_indices, np.arange(probabilities.shape[1])] = 1
        
        return predictions
        ######################################################################

    def backward(
        self, 
        X: npt.ArrayLike, 
        Y: npt.ArrayLike, 
        learning_rate: float = 0.005
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Backpropagation algorithm.
        """
        ########################## STUDENT SOLUTION ###########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Perform forward pass, then backpropagation
        #         to get gradient for weight matrices and biases
        #         2) Return the gradient for weight matrices and biases
        
        
        # perform forward pass
        z1 = np.dot(self.weights_1, X) + self.bias_1
        a1 = relu(z1)
        z2 = np.dot(self.weights_2, a1) + self.bias_2
        a2 = softmax(z2)
                
        # initialize the total loss 
        total_loss = compute_loss(a2, Y)
        
        # Backpropagation
                
        # calculate the loss for output and hidden layer
        loss_2 = a2 - Y
        loss_1 = np.multiply(np.dot(self.weights_2.T, loss_2), relu_prime(z1))
        

        # calculate the partial derivative of the loss with respect to the weights in both the layers
        # the partial derivative with respect to the bias is equal to the loss of the layers        
        delta_w2 = np.dot(loss_2, a1.T)
        delta_b2 = np.sum(loss_2, axis=1, keepdims=True) / X.shape[0]
        delta_w1 = np.dot(loss_1, X.T)
        delta_b1 = np.sum(loss_1, axis=1, keepdims=True) / X.shape[0]
        
        # update weights and biases using gradients and learning rate
        
        self.weights_2 -= learning_rate * (delta_w2/X.shape[0])
        self.bias_2 -= learning_rate * (delta_b2/X.shape[0])
        self.weights_1 -= learning_rate * (delta_w1/X.shape[0])
        self.bias_1 -= learning_rate * (delta_b1/X.shape[0])
        
        print("The average loss now is: ", total_loss)      
        
        # return self.weights_2, self.bias_2, self.weights_1, self.bias_1
        return total_loss
        #######################################################################


def compute_loss(pred: npt.ArrayLike, truth: npt.ArrayLike) -> float:
    """
    Compute the cross entropy loss.
    """
    ########################## STUDENT SOLUTION ###########################
    # YOUR CODE HERE
    #     TODO:
    #         1) Compute the cross entropy loss between your model prediction
    #         and the ground truth.
    #return -np.sum(np.log(pred[np.arange(len(truth))])) / len(truth)
    return -np.sum(truth * np.log(pred + 1e-15)) / len(truth[0])
    #######################################################################

# in_weights = NeuralNetwork(X.shape[0], 150, Y.shape[0]).forward(X)[0]
# in_bias = NeuralNetwork(X.shape[0], 150, Y.shape[0]).forward(X)[1]
# out_weights = NeuralNetwork(X.shape[0], 150, Y.shape[0]).forward(X)[2]
# out_bias = NeuralNetwork(X.shape[0], 150, Y.shape[0]).forward(X)[3]

# Y_pred = NeuralNetwork(X.shape[0], 150, Y.shape[0]).predict(X)

# probabilities = NeuralNetwork(X.shape[0], 150, Y.shape[0]).forward(X)

# loss = compute_loss(Y_pred, Y)
# print(loss)
