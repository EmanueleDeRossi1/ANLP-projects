from collections import defaultdict
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class NaiveBayes(object):

    ######################### STUDENT SOLUTION #########################
    # YOUR CODE HERE
    def __init__(self, class_probs, feature_probs):
        """Initialises a new classifier."""
        self.class_probs = class_probs
        self.feature_probs  = feature_probs
    ####################################################################


    def predict(self, x):
        """Predicts the class for a document.

        Args:
            x: A document, represented as a list of words.

        Returns:
            The predicted class, represented as a string.
        """
        ################## STUDENT SOLUTION ########################
        # YOUR CODE HERE
        labels = set(self.class_probs)

        probabilities_dict = {}




        for label in labels:
            current_label_probability = np.log(self.class_probs[label])
            #current_label_probability = self.class_probs[label]

            
            for word in x:
                if word in self.feature_probs[label]:
                    current_label_probability = current_label_probability + np.log(self.feature_probs[label][word])
            probabilities_dict[label] = current_label_probability 
                    #print(probabilities_dict)
                    #print(current_label_probability)
        #print(probabilities_dict)
        # if max(probabilities_dict) != "offensive":
        #     print(x)
        # print("highest", max(probabilities_dict.values()))
        # print(probabilities_dict)
        
        return max(probabilities_dict, key=probabilities_dict.get)
        ############################################################


    @classmethod
    def train(cls, data, k=1):
        """Train a new classifier on training data using maximum
        likelihood estimation and additive smoothing.

        Args:
            cls: The Python class representing the classifier.
            data: Training data.
            k: The smoothing constant.

        Returns:
            A trained classifier, an instance of `cls`.
        """
        ##################### STUDENT SOLUTION #####################
        # YOUR CODE HERE
        #classifier = cls()

        class_probs = {}
        feature_probs = defaultdict(dict)


        tot_num_docs = len(data)
        #tot_num_features = len(set(feature for instance in data for feature in instance[0]))

        class_count = defaultdict(int)
        feature_count = defaultdict(lambda: defaultdict(int))

        # count number occurences
        for instance in data:
            label = instance[1]
            class_count[label] += 1

            for feature in instance[0]:
                feature_count[label][feature] += 1
                
        # extract vocabulary of words
        vocabulary = set(feature for label in feature_count.values() for feature in label.keys())
        #print("vocabulary", vocabulary)
        total_words = sum([len(instance[0]) for instance in data])
        print(len(vocabulary))
        #print(total_words)

        # vocabulary_sum  = sum(inner_value for inner_dict in feature_count.values() for inner_value in inner_dict.values())
        # print(vocabulary_sum)

        # calculate probabilities
        for label in class_count:
            class_probs[label] = class_count[label] / tot_num_docs
            total_word_count = sum(feature_count[label].values())
            for feature in vocabulary:
                if feature not in feature_count[label]:
                    feature_count[label][feature] = 0
            for feature in feature_count[label]:
                feature_probs[label][feature] = (feature_count[label][feature] + k) / (total_word_count + (k * len(vocabulary)))
            # print("sum values of label: ", label, sum(feature_probs[label].values()))
            
        return cls(class_probs, feature_probs)
        # return vocabulary
        ############################################################


def features1(data, k=1):
    """
    Your feature of choice for Naive Bayes classifier.

    Args:
        data: Training data.
        k: The smoothing constant.

    Returns:
        Parameters for Naive Bayes classifier, which can
        then be used to initialize `NaiveBayes()` class
    """
    ###################### STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    # perform stemmetization on the dataset
    
    ps = PorterStemmer()
    stemmed_data = []
    
    for instance in data:
        label = instance[1]
        stammed_tweet = []
        for word in instance[0]:
            stammed_tweet.append(ps.stem(word))
        stemmed_data.append((stammed_tweet, label))

            
        

        
    return stemmed_data
    ##################################################################

def features2(data, k=1):
    """
    Your feature of choice for Naive Bayes classifier.

    Args:
        data: Training data.
        k: The smoothing constant.

    Returns:
        Parameters for Naive Bayes classifier, which can
        then be used to initialize `NaiveBayes()` class
    """
    
    ###################### STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    
    # remove stop words
    stop_words = set(stopwords.words("english"))
    data_without_stop = []
    for instance in data:
        label = instance[1]
        tweet_without_stop = []
        for word in instance[0]:
            if word.lower() not in stop_words:
                tweet_without_stop.append(word)
        data_without_stop.append((tweet_without_stop, label))
    

    return data_without_stop
    
    ##################################################################

