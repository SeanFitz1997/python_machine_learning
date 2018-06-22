#Imports=======================================================================================================================
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#Classes=======================================================================================================================
class Pertcetron(object):
    """
    Perceptron Classisfier
    Parameters:
    learningRate: float (between 0.0 and 1.0)
    epochs: int (The number of passes ofver the training dataset)
    Attributes:
    weights_: 1D array (The weights after training)
    errors_: The number of misclassifications
    """
    def __init__(self, learningRate = 0.01, epochs = 10):
        self.learningRate = learningRate
        self.epochs = epochs

    def net_input(self, X):
        """Caluclate net input"""
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def predict(self, X):
        """Reteurn class lable after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1 , -1)

    def fit(self, X, y):
        """
        Fit training data
        Parameters:
        X: array , shape = [n_samples,n_features] 
        (training vectors where:
            n_samples is the number of training samples,
            n_features is the number of features)
        y: 1D array in the form [n_samples] (training lables)
        Returns: self: object
        """
        self.weights_= np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors=0    
            for xi, target in zip(X, y):   
                update = self.learningRate * (target - self.predict(xi))
                self.weights_[1:] += update * xi
                self.weights_[0] += update
                errors += int(update != 0.0) 
            self.errors_.append(errors)
        return self      
#Functions======================================================================================================================
def plot_training_data(X):
    """ Description: Draws a scatter plot of the training data before treshold level is found
        Parameters: X (Dataframe) Training data
    """
    plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
    plt.scatter(X[50:, 0], X[50:, 1], color = 'blue', marker = 'x', label = 'versicolor')
    plt.legend(loc = "upper left")
    plt.xlabel("septal length [cm]")
    plt.ylabel("petal length [cm]")
    plt.show()

def plot_errors_per_epoch(classifier):
    """ Description: Draws a line graph of the number of errors at each epoch
        Parameters: classifier (Pertcetron)
    """
    plt.plot(range(1, len(classifier.errors_) + 1), classifier.errors_, marker = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()

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
    plt.xlabel("septal length [cm]")
    plt.ylabel("petal length [cm]")
    plt.legend(loc = "upper left")
    plt.show()

#Program========================================================================================================================          
usageString = 'Usage:\nFor help:\tperceptron_learning_algorithm.py -help\nPlot training data:\tperceptron_learning_algorithm.py -inputData\nPlot errors at each epoch\tperceptron_learning_algorithm.py -errors\nPlot decision region:\tperceptron_learning_algorithm.py -trainedData\nTo input data:\tperceptron_learning_algorithm.py [septal length (cm)] [petal length (cm)]'
if("-help" in sys.argv):    
    print(usageString)
    sys.exit()
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../Percerptron_Learning_Algorithm(Python)/iris.csv"), header = None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values
if("-inputData" in sys.argv):    
    plot_training_data(X)
    sys.exit()
ppn = Pertcetron(learningRate = 0.1, epochs = 10) #Create Perceptron Model
ppn.fit(X, y)
if("-errors" in sys.argv):    
    plot_errors_per_epoch(ppn)
    sys.exit()
if("-trainedData" in sys.argv):   
    plot_decision_regions(X, y, classifier = ppn)
    sys.exit()
#Find class of inputed data
if(len(sys.argv) < 3):
    print("Error: Too few arguments passed")
    print(usageString)
    sys.exit()
elif(len(sys.argv) > 3):
    print("Error: Too many arguments passed")
    print(usageString)
    sys.exit()
else:
    try:
        sepLen = float(sys.argv[1])
        petLen = float(sys.argv[2])
    except ValueError:
        print("Error: Invailid inputs. Values must be floats")
        print(usageString)
        sys.exit()
    inputVal = np.array([sepLen, petLen])
    if(ppn.predict(inputVal) == -1):    print("Iris-setosa")
    else:   print("Iris-versicolor")