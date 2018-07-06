#Imports===================================
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import sys

#Functions=================================
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.2):
    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #plot decision region
    x1_min, x_1max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x_1max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(x1_min, x_1max)
    plt.ylim(x2_min, x2_max)
    #plot samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
#Program===================================
usageString = 'Usage:\nFor help:\tSVM.py -help\nPlot decision region:\tSVM.py -trainedData\nTo input data:\tSVM.py [petal length (cm)] [petal width (cm)]'
if("-help" in sys.argv):    
    print(usageString)
    sys.exit()

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
#split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#standardize features
sc = StandardScaler()
sc.fit(X_train) #estimates the sample mean and standard deviation for each feature
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined_std = np.hstack((y_train, y_test))
#Create & train SVM
svm = SVC(kernel='linear', C=1, random_state=0)
svm.fit(X_train_std, y_train)

if("-trainData" in  sys.argv):
    plot_decision_regions(X_combined_std, y_combined_std, classifier=svm, test_idx=range(105,150))
    plt.xlabel('petal length [standardised]')
    plt.ylabel('petal width [standardised]')
    plt.legend(loc='upper left')
    #rename leg labels
    L=plt.legend()
    L.get_texts()[0].set_text('Iris-setosa')
    L.get_texts()[1].set_text('Iris-versicolor')
    L.get_texts()[2].set_text('Iris-virginica')
    plt.show()
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
        petLength = float(sys.argv[1])
        petWidth = float(sys.argv[2])
    except ValueError:
        print("Error: Invailid inputs. Values must be floats")
        print(usageString)
        sys.exit()
    inputVal = np.array([petLength, petWidth]).reshape(1, -1)
    inputVal = sc.transform(inputVal)   #standardise input values   
    if(svm.predict(inputVal) == 0):    print("Iris-setosa")
    elif(svm.predict(inputVal) == 0):    print("Iris-versicolor")
    else: print('Iris-virginica')