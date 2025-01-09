import numpy as np
import numpy.typing as npt
from collections import Counter
import string

from typing import Tuple, List, Set


def bag_of_words_matrix(sentences: List[str]) -> npt.ArrayLike:
    """
    Convert the dataset into V x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    # Create translation table to remove punctuation
    translator = str.maketrans("", "", string.punctuation)

    # collect all words into a single list and all sentences into a single list
    word_list = [word for sentence in sentences for word in sentence.translate(translator).split()]
    sentence_list = [sentence.translate(translator).split() for sentence in sentences]
    
    # count the occurances of the words
    word_count = Counter(word_list)
    # if a word appears less than twice, ignore it
    unique_words = [word for word, count in word_count.items() if count <= 2]
    
    # create matrix
    X = np.zeros((len(unique_words),len(sentences)))
    
    
    # create a bag of words
    for index_sentence, sentence in enumerate(sentence_list):
        for word in sentence:
            if word in unique_words:
                X[unique_words.index(word)][index_sentence] = 1
                #print(word, index)
    
    # YOUR CODE HERE
    return X
    #########################################################################


def labels_matrix(data: Tuple[List[str], Set[str]]) -> npt.ArrayLike:
    """
    Convert the dataset into K x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    sentences = data[0]
    intents = data[1]
    
    unique_intents = list(set(intents))
    
    Y = np.zeros((len(unique_intents), len(sentences)))
    
    for sentence, intent in zip(sentences, intents):
        Y[unique_intents.index(intent)][sentences.index(sentence)] = 1
    return Y
    #########################################################################


def softmax(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Softmax function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    exp_logits = np.exp(z)  
    return exp_logits/sum(exp_logits)
    #########################################################################


def relu(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Rectified Linear Unit function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    return np.maximum(0, z)
    #########################################################################


def relu_prime(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    First derivative of ReLU function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    return np.where(z < 0, 0, 1)
    #########################################################################
    
