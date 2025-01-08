import matplotlib.pyplot as plt
import numpy as np

from model.naivebayes import NaiveBayes, features1, features2
from model.logreg import LogReg, featurize
from evaluation import accuracy, f_1


def train_smooth(train_data, test_data):
                
    # YOUR CODE HERE
    #     TODO:
    #         1) Re-train Naive Bayes while varying smoothing parameter k,
    #         then evaluate on test_data.
    #         2) Plot a graph of the accuracy and/or f-score given
    #         different values of k and save it, don't forget to include
    #         the graph for your submission.

    ######################### STUDENT SOLUTION #########################
    k_values = [0.01, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5]
    accuracies = []
    f_1s = []
    
    for k_value in k_values:
        nb = NaiveBayes.train(train_data, k=k_value)
        acc = accuracy(nb, test_data)
        f1 = f_1(nb, test_data)
        accuracies.append(acc)
        f_1s.append(f1)    
        #print("accuracy with k value", k_value, accuracy(nb, test_data)) 
        #print("f-1 score with k_value: ", k_value, f_1(nb, test_data))
        
    #x_ticks = np.arange(len(k_values))

    plt.figure()


    plt.subplot(1,2,1)
    plt.bar(k_values, accuracies, width=0.2, color="red")
    #plt.xticks(x_ticks, k_values)  # Set x-axis ticks

    plt.subplot(1,2,2)
    plt.bar(k_values, f_1s, width=0.2, color="green")
    #plt.xticks(x_ticks, k_values)  # Set x-axis ticks
    #plt.tight_layout()
    plt.show()


    
    #return accuracies, f_1s
    ####################################################################



def train_feature_eng(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Improve on the basic bag of words model by changing
    #         the feature list of your model. Implement at least two
    #         variants using feature1 and feature2
    ########################### STUDENT SOLUTION ########################
    no_stop_stem_train_data = features1(features2(train_data))
    no_stop_stem_test_data = features1(features2(test_data))
    nb = NaiveBayes.train(no_stop_stem_train_data)
    print("Accuracy on stemmed data with removed stop words: ", accuracy(nb, no_stop_stem_test_data))
    print("F_1 on stemmed data with removed stop words: ", f_1(nb, no_stop_stem_test_data))

    
    
    #####################################################################



def train_logreg(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) First, assign each word in the training set a unique integer index
    #         with `buildw2i()` function (in model/logreg.py, not here)
    #         2) Now that we have `buildw2i`, we want to convert the data into
    #         matrix where the element of the matrix is 1 if the corresponding
    #         word appears in a document, 0 otherwise with `featurize()` function.
    #         3) Train Logistic Regression model with the feature matrix for 10
    #         iterations with default learning rate eta and L2 regularization
    #         with parameter C=0.1.
    #         4) Evaluate the model on the test set.
    ########################### STUDENT SOLUTION ########################
    pass
    #####################################################################
