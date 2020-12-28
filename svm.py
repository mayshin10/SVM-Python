import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt

# file path
iris = "iris_dataset.csv"

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
function description
name : data_preprocessing
argument : file path of iris
I utilized two properties of flowers; petal length, and petal width.
The function read data from csv file and slicing it to appropriate data.

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def data_preprocessing(file):
    '''original csv data'''
    data_origin = pd.read_csv(iris).to_numpy()

    '''setosa data related to petal_length and petal_width'''
    setosa_train = np.hstack([data_origin[:, 2:4], data_origin[:, 4:5]])[0:40]
    setosa_test = np.hstack([data_origin[:, 2:4], data_origin[:, 4:5]])[40:50]

    '''versicolor data related to petal_length and petal_width'''
    versicolor_train = np.hstack([data_origin[:, 2:4], data_origin[:, 4:5]])[50:90]
    versicolor_test = np.hstack([data_origin[:, 2:4], data_origin[:, 4:5]])[90:100]

    '''virginica data related to petal_length and petal_width'''
    virginica_train = np.hstack([data_origin[:, 2:4], data_origin[:, 4:5]])[100:140]
    virginica_test = np.hstack([data_origin[:, 2:4], data_origin[:, 4:5]])[140:150]

    '''combine'''
    train = np.vstack([setosa_train, versicolor_train, virginica_train])
    test = np.vstack([setosa_test, versicolor_test, virginica_test])
    return train, test


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
function description
name : svm_fitting
argument : 1. train data,  2. the first flower to compare, 3. the second flower to compare
This function makes a model using svm algorithm.
Create a straight line that separates the two data, 
and create a straight line with the widest margin through the gradient descent algorithm.

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def svm_fitting(train_data, flower1, flower2):
    '''train set'''
    X = np.array(train_data[:, 0:2], dtype=np.float64)
    '''ground truth'''
    Y = []
    for j in train_data[:, 2]:
        if j == flower1:
            Y.append(-1)
        elif j == flower2:
            Y.append(1)

    '''initialize w and b'''
    np.random.seed(77)
    w_train = np.random.randn(2, 1)
    b_train = np.random.randn(1)

    '''loop adjusting w and b'''
    for i in range(1000):
        '''call fit function'''
        w_train, b_train = fit(w_train, b_train, X)

        '''prediction stage'''
        prediction = []
        pred = w_train.T.dot(X.T) + b_train
        for j in range(len(X)):
            if pred[0, j] < -1:
                prediction.append(-1)
            else:
                prediction.append(1)

        '''adjusting stage'''
        sum = 0
        for z in range(len(X)):
            if Y[z] != prediction[z]:
                sum = sum + 1
                '''cost function = Sum of ((-target value)*(w^T * x_k +b))'''
                w_train = w_train + 0.002 * (Y[z] * X[z: z + 1, :].T)
                b_train = b_train + 0.002 * (Y[z])

    return w_train, b_train


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
function description
name : fit
argument : 1. weight,  2. bias, 3. train data
The function receives weight and bias input.
This function changes the value so that weight and bias meet the following expression.
|w^T * x_k +b| =1

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def fit(w_fit, b_fit, X):
    '''distance array'''
    h = np.abs(w_fit.T.dot(X.T) + b_fit) / np.sqrt(w_fit.T.dot(w_fit))
    '''inner product with support vector'''
    A = np.abs(w_fit.T.dot(X[h.argmin(), :].T) + b_fit)
    '''tuning parameters'''
    w_fit = w_fit / A
    b_fit = b_fit / A
    return w_fit, b_fit



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
function description
name : svm_prediction
argument : 1. weight,  2. bias, 3. test data, 4. the first flower to compare, 5. the second flower to compare
test results and show accuracy.


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def svm_prediction(w_pred, b_pred, test_pred, flower1, flower2):
    result = []
    '''calculate |w^T * x_k +b| and evaluate '''
    for i in range(len(test_pred)):
        if w_pred.T.dot(test_pred[i, 0:2].T) + b_pred <= -1:
            result.append(flower1)
        elif w_pred.T.dot(test_pred[i, 0:2].T) + b_pred >= 1:
            result.append(flower2)
        else:
            result.append('error')

    '''calcuate accuracy'''
    accuracy = 0
    for i in range(len(test_pred)):
        if test_pred[i, 2] == result[i]:
            accuracy = accuracy + 1
    accuracy = accuracy / len(test_pred) * 100

    print(flower1, " and ",flower2, " Accuracy : ", accuracy , "%")

    return accuracy


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
function description
name : svm_visualization
argument : 1. weight1,  2. bias1, 3. weight1, 4. bias2, 5. train 6. test
The function implements visualizing results and draw the hyperplane

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def svm_visualization(w1_visual, b1_visual, w2_visual, b2_visual, train_visual, test_visual):
    '''hyperplane between setosan and versicolor'''
    x1 = np.arange(1, 8)
    y1 = -w1_visual[0, 0] / w1_visual[1, 0] * x1 - b1_visual / w1_visual[1, 0]

    '''hyperplane between versicolor and virginica'''
    x2 = np.arange(1, 8)
    y2 = -w2_visual[0, 0] / w2_visual[1, 0] * x2 - b2_visual / w2_visual[1, 0]

    plt.plot(x1, y1, color='r')
    plt.plot(x2, y2, color='b')

    '''plot the train and test data with scatter manner'''
    plt.scatter(train_visual[0:40, 0], train_visual[0:40, 1], label='setosa_train',marker='o',  c='red')
    plt.scatter(train_visual[40:80, 0], train_visual[40:80, 1], marker='x', label='versicolor_train', c='blue')
    plt.scatter(train_visual[80:120, 0], train_visual[40:80, 1], marker='x', label='verginica_train', c='black')

    plt.scatter(test_visual[0:10, 0], test_visual[0:10, 1], marker='o', label='setosa_test', c='magenta')
    plt.scatter(test_visual[10:20, 0], test_visual[10:20, 1], marker='x', label='versicolor_test', c='cyan')
    plt.scatter(test_visual[20:30, 0], test_visual[20:30, 1], marker='x', label='virginica_test', c='green')

    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc='best')
    plt.show()

    return 0


'''implement a total of two SVM algorithms'''
train_data, test_data = data_preprocessing(iris)
w1, b1 = svm_fitting(train_data[0:80, :], 'setosa', 'versicolor')
accuracy1 = svm_prediction(w1, b1, test_data[0:20, :], 'setosa', 'versicolor')
w2, b2 = svm_fitting(train_data[40:120, :], 'versicolor', 'virginica')
accuracy2 = svm_prediction(w2, b2, test_data[10:30, :], 'versicolor', 'virginica')

'''show the final accuracy'''
print("Total SVM accuracy = ", accuracy1/2 + accuracy2/2, "%")
'''show the final result'''
svm_visualization(w1, b1, w2, b2, train_data, test_data)
