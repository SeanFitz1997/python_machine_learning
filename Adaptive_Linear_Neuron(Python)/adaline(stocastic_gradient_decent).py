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
    ADAptive LInear NEuron Classisfier, with Stocasti Gradient Decent
    Parameters:
    learningRate: float (between 0.0 and 1.0)
    epochs: int (The number of passes ofver the training dataset)
    Attributes:
    weights_: 1D array (The weights after training)
    errors_: The number of misclassifications
    shuffle: bool Shuffles the training set after each epoch to prevent cycles (default: True)
    random_state: int Set a random state for shuffleing and initalising weights (default: None)
    """
    def __init__(self, learningRate = 0.01, epochs = 10, shuffle = True, random_state = None):
        self.learningRate = learningRate
        self.epochs = epochs
        self.weights_initalised = False
        self.shuffle = True

        if(random_state):
            np.random.seed(random_state)

    def net_input(self, X):
        """ Caluclate net input """
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def activation(self, X):
        """ Computes linear activation  """
        return self.net_input(X)


    def predict(self, X):
        """Reteurn class lable after unit step"""
        return np.where(self.activation(X) >= 0.0, 1 , -1)

    def _shuffle(self, X, y):
        """ Shuffle training data """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initalise_weights(self, m):
        """ Initialise weights to zero """
        self.weights_= np.zeros(1 + m)
        self.weights_initalised = True

    def _update_weights(self, xi, target):
        """ Apply adaline learning rule to update the weights """
        output = self.net_input(xi)
        error = (target - output)
        self.weights_[1:] += self.learningRate * xi.dot(error)
        self.weights_[0] += self.learningRate * error
        cost = 0.5 * error**2
        return cost

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
        self._initalise_weights(X.shape[1])
        self.cost_ = []

        #stocastic gradient decent
        for i in range(self.epochs):
            if(self.shuffle):
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """ Fit training data without reinitializing the weights """
        if not(self.weights_initalised):
            self._initalise_weights(X.shape[1])
        if(y.ravel().shape[0] > 1):
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X,y)
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
    """ Description: Draws a line graph showing the cost curve at each epoch"""
    ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 4))
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

def plot_cost_function_with_stadardised_data(X, y, learningRate):
    """ Description: Draws a line graph showing the cost curve at each epoch    
        * Takes original data and will standardise it   *"""
    #Standardising data
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    #Adaline with feature scaling
    ada = AdalineGD(epochs=15, learningRate=learningRate)
    ada.fit(X_std, y)
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Sum-squared-errors')
    plt.show()


#Program========================================================================================================================
usageString = 'Usage:\nFor help:\tadaline.py -help\nCompare Learning rates:\tadaline.py -learningRates [rate 1] [rate 2]\nPlot decision region:\tadaline.py -trainedData\nPlot cost function standardised:\tadaline.py -cost_Function_Standardised [learing rate]\nTo input data:\tadaline.py [septal length (cm)] [petal length (cm)]'
if("-help" in sys.argv):    
    print(usageString)
    sys.exit()
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../Percerptron_Learning_Algorithm(Python)/iris.csv"), header = None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values
#Standardising data
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
ada = AdalineGD()
ada.fit(X_std, y)

if("-learningRates" in sys.argv):
    try:
        rate1 = float(sys.argv[2])
        rate2 = float(sys.argv[3])
    except ValueError:
        print("Error: invalid input")
        print(usageString)
        sys.exit()
    print('r1 = ' + str(rate1) + ' r2 = ' + str(rate2))
    if(rate1 > 1 or rate2 > 1 or rate1 < 0 or rate2 < 0):
        print("Error: invalid input. Values must be between 0 and 1")
        print(usageString)
        sys.exit()
    plot_cost_function_at_different_learning_rates(X,y,rate1,rate2)
elif("-trainedData" in sys.argv):
    plot_decision_regions(X_std, y, ada)
    sys.exit()
elif("-cost_Function_Standardised" in sys.argv):
    try:
        learningRate = float(sys.argv[2])
    except ValueError:
        print("Error: Invailid input. Value must be floats")
        print(usageString)
        sys.exit()
    plot_cost_function_with_stadardised_data(X, y, learningRate)

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
    if(ada.predict(inputVal) == -1):    print("Iris-setosa")
    else:   print("Iris-versicolor")
