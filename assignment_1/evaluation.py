def accuracy(classifier, data):
    """Computes the accuracy of a classifier on reference data.

    Args:
        classifier: A classifier.
        data: Reference data.

    Returns:
        The accuracy of the classifier on the test data, a float.
    """
    ##################### STUDENT SOLUTION #########################
    
    num_correct = 0
    for instance in data:
        #print(instance)
        prediction = classifier.predict(instance[0])
        # print(prediction)
        true_label =  instance[1]
        if prediction == true_label:
            num_correct += 1
    acc = num_correct/len(data)
    
    return acc 


    ################################################################



def f_1(classifier, data):
    """Computes the F_1-score of a classifier on reference data.

    Args:
        classifier: A classifier.
        data: Reference data.

    Returns:
        The F_1-score of the classifier on the test data, a float.
    """
    ##################### STUDENT SOLUTION #########################
    # YOUR CODE HERE
    
    tp = 0.0
    fp = 0.0
    fn = 0.0
    
    for instance in data:
        prediction = classifier.predict(instance[0])
        true_label = instance[1]
        if prediction == true_label and prediction == "offensive":
            tp += 1
        elif prediction != true_label and prediction == "offensive":
            fp += 1
        elif prediction != true_label and prediction == "nonoffensive":
            fn += 1
            
    # print("tp", tp)
    # print("fp", fp)
    # print("fn", fn)
    
    precision = tp / (tp + fp)
    # print("precision", precision)
    recall = tp / (tp + fn)
    # print("recall", recall)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score
            

    ################################################################

### test your functions

# import numpy as np

# test_data = [1, 1, 0, 1, 0]

# def test_predict(input_data):
    
#     predictions = []
    
#     for i in input_data:
#         pred = np.random.choice([0, 1])
#         predictions.append(pred)
        
#     return predictions

# test_predictions = test_predict(test_data)

# print("predictions", test_predictions)
# print("true labels", test_data)

# #print(accuracy(test_predictions, test_data))
# print(f_1(test_predictions, test_data))