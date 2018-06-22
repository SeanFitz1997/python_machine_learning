# Percerptron_Learning_Algorithm(Python)
Percptron classifier determines if a flower is **Iris-setosa** or **Iris-versicolor** based on its inputed **septal length [cm]** and **petal length [cm]**.
## Requirments
You must have installed the **numpy**, **pandas** and **matplotlib** modules
## Setup
To start clone this repo and move to this directory:
```
git clone https://github.com/SeanFitz1997/python_machine_learning.git
cd Percerptron_Learning_Algorithm(Python)
```
## Usage
- For help:       perceptron_learning_algorithm.py -help
- Plot training data:     perceptron_learning_algorithm.py -inputData
- Plot errors at each epoch       perceptron_learning_algorithm.py -errors
- Plot decision region:   perceptron_learning_algorithm.py -trainedData
- To input data:  perceptron_learning_algorithm.py [septal length (cm)] [petal length (cm)]
## Example
(input data)
```
python perceptron_learning_algorithm.py 4.9 3
```
Expected output
```
Iris-versicolor
```
