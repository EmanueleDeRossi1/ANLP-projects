import numpy as np


class LogReg:
    def __init__(self, eta=0.01, num_iter=30):
        self.eta = eta
        self.num_iter = num_iter

    def softmax(self, inputs):
        """
        Calculate the softmax for the given inputs (array)
        :param inputs:
        :return:
        """
        # TODO: adapt for your solution
        # return np.exp(inputs) / float(sum(np.exp(inputs)))
        return np.exp(inputs) / np.sum(np.exp(inputs), axis=1, keepdims=True)



    def train(self, X, Y):

        #################### STUDENT SOLUTION ###################

        # weights initialization
        self.weights = np.zeros((X.shape[1], Y.shape[1])) 
        for i in range(self.num_iter):
            z = np.dot(X, self.weights)
            probabilities = self.softmax(z)
            
            loss = Y - probabilities
            gradient = np.dot(X.T, loss)
            
            self.weights = self.weights + (self.eta * gradient)
            
            # YOUR CODE HERE
            #     TODO:
            #         1) Fill in iterative updating of weights
            
        return self
        #########################################################


    def p(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Fill in (log) probability prediction
        ################## STUDENT SOLUTION ########################
        z = np.dot(X, self.weights)
        return self.softmax(z)

        ############################################################


    def predict(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Replace next line with prediction of best class
        ####################### STUDENT SOLUTION ####################
        probabilities = self.p(X)
        return np.argmax(probabilities, axis=1)
        #############################################################


def buildw2i(vocab):
    """
    Create indexes for 'featurize()' function.

    Args:
        vocab: vocabulary constructed from the training set.

    Returns:
        Dictionaries with word as the key and index as its value.
    """
    # YOUR CODE HERE
    #################### STUDENT SOLUTION ######################
    
    word_dictionaries = dict()
    
    for idx, word in enumerate(vocab):
        word_dictionaries[word] = idx
    return word_dictionaries
    ############################################################


def featurize(data, train_data=None):
    """
    Convert data into X and Y where X is the input and
    Y is the label.

    Args:
        data: Training or test data.
        train_data: Reference data to build vocabulary from.

    Returns:
        Matrix X and Y.
    """
    # YOUR CODE HERE
    ##################### STUDENT SOLUTION #######################
    vocab = set([word.lower() for instance in train_data for word in instance[0]])
    word_dictionaries = buildw2i(vocab)
    X = np.zeros((len(data), len(word_dictionaries)))
    Y = np.zeros((len(data), 2))
    
    for instance_idx, instance in enumerate(data):
        # create X matrix
        for word in instance[0]:
            if word.lower() in word_dictionaries:
                X[instance_idx, word_dictionaries[word.lower()]] = 1
        # create Y matrix
        if instance[1] == "offensive":
            Y[instance_idx, 0] = 1
        else:
            Y[instance_idx, 1] = 1
    
    
    return X, Y
    ##############################################################
