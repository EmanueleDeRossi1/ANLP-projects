from model.ffnn import NeuralNetwork, compute_loss
import matplotlib.pyplot as plt

def batch_train(X, Y, model, train_flag=False):
    ################################# STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) Use your neural network to predict the intent
    #         (without any training) and calculate the accuracy 
    #         of the classifier. Should you be expecting high
    #         numbers yet?
    #         2) if train_flag is true, run the training for 1000 epochs using 
    #         learning rate = 0.005 and use this neural network to predict the 
    #         intent and calculate the accuracy of the classifier
    #         3) Then, plot the cost function for each iteration and
    #         compare the results after training with results before training
    prediction = model.predict(X)
    print("The loss of the model without any training is:", compute_loss(prediction, Y))
    if train_flag:
        n_epochs = 1000
        loss = []
        for epoch in range(n_epochs):
            loss_epoch = model.backward(X, Y)
            print(loss_epoch, "number of epochs: ", epoch)
            loss.append(loss_epoch)
        plt.plot(loss)
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
    ###############################################################################


def minibatch_train(X, Y, model, train_flag=False):
    ########################## STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) As bonus, train your neural network with batch size = 64
    #         and SGD (batch size = 1) for 1000 epochs using learning rate
    #         = 0.005. Then, plot the cost vs iteration for both cases.
    pass
    if train_flag:
        pass
    #########################################################################
