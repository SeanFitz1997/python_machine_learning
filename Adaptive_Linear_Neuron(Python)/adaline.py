#Imports=======================================================================================================================
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#Classes=======================================================================================================================
class AdalineGD(object):
    """
    ADAptive LInear NEuron Classisfier
    Parameters:
    learningRate: float (between 0.0 and 1.0)
    epochs: int (The number of passes ofver the training dataset)
    Attributes:
    weights_: 1D array (The weights after training)
    errors_: The number of misclassifications
    """
    def __init__(self, learningRate = 0.01, epochs = 50):
        self.learningRate = learningRate
        self.epochs = epochs

    def net_input(self, X):
        """ Caluclate net input """
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def activation(self, X):
        """ Computes linear activation  """
        return self.net_input(X)


    def predict(self, X):
        """Reteurn class lable after unit step"""
        return np.where(self.activation(X) >= 0.0, 1 , -1)

    def fit(self, X, y):
        """
        Fit training data
        Parameters:
        X: array , shape = [n_samples, n_features] 
        (training vectors where:
            n_samples is the number of training samples,
            n_features is the number of features)
        y: 1D array in the form [n_samples] (target values)
        Returns: self: object
        """
        self.weights_= np.zeros(1 + X.shape[1])
        self.cost_ = []

        #Instead of updating the weights after each training sample(perecptron), 
        #we calculate the gradient bases on the entire training set
        for _ in range(self.epochs):
            output = self.net_input(X)
            errors = (y - output)   
            self.weights_[1:] += self.learningRate * X.T.dot(errors)    #for weights 1 - m (matrix multiplication between feature matrix and error vector)
            self.weights_[0] += self.learningRate * errors.sum()    #for [0] weight
            cost = (errors ** 2).sum() / 2
            self.cost_.append(cost) #collect the costs to see if the algorithm converged after training
        return self

#Functions======================================================================================================================
def plot_decision_regions(X, y, classifier, resolution = 0.02):
    """ Description: Draws a sctter plot of training samples in regions seperated by the threshold line
        Parameters:
        X (2D array) training data
        y (1D array) training lables
        classifier (AdalineGD)
        resolution (float)        """
    #setup marker generaton and color map
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','grey','cryan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #plot decision suface
    x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max() + 1 #Find the min and max values for the two features
    x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)) #An vector array for the values of the 2 features
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) #ravel() flattens the grid arrays to create a matrix that has to same number 0of cols as the iris training set
    z = z.reshape(xx1.shape) #reshape predict class lables into a grid
    plt.contourf(xx1, xx2, z, alpha = 0.4, cmap = cmap) #Draws a contorf plot
    plt.xlim(xx1.min(), xx1.max()) 
    plt.ylim(xx2.min(), xx2.max())
    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        if(cl == -1):   lable = 'Iris-setosa'
        else:   lable = 'Iris-versicolor'
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha = 0.8, c = cmap (idx), marker = markers[idx], label = lable)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardised]')
    plt.ylabel('petal length [standardised]')
    plt.legend(loc='upper left')
    plt.show()

def plot_cost_function_at_different_learning_rates(X, y, rate1 = 0.01, rate2 = 0.0001):
    """ Description: TODO"""
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 4))
    # Learing rate 1
    ada1 = AdalineGD(epochs = 10, learningRate = rate1).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-errors)')
    ax[0].set_title('Adaline - Learning rate = ' + str(rate1))
    # Learning rate 2
    ada2 = AdalineGD(epochs = 10, learningRate = rate2).fit(X, y)
    ax[1].plot(range(1,len(ada2.cost_) +1), np.log10(ada2.cost_), marker = 'o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Sum-squared-errors)')
    ax[1].set_title('Adaline - Learning rate = ' + str(rate2))

    plt.show()

def plot_cost_function_with_stadardised_data(X, y):
    """ Description: TODO"""
    #Standardising data
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    #Adaline with feature scaling
    ada = AdalineGD(epochs=15, learningRate=0.01)
    ada.fit(X_std, y)
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Sum-squared-errors')
    plt.show()


#Program========================================================================================================================
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../Percerptron_Learning_Algorithm(Python)/iris.csv"), header = None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values

plot_cost_function_at_different_learning_rates(X, y, 0.0001, 0.0001)

