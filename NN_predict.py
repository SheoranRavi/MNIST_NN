import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import NN_Backprop_fns as NN
import MNIST_normalize as Norm

def predict(X, Theta1, Theta2, m):
    X = np.c_[np.ones(m), X]
    z2 = np.dot(X, Theta1.T)
    A2 = NN.sigmoid(z2)
    A2 = np.c_[np.ones(m), A2]
    z3 = np.dot(A2, Theta2.T)
    A3 = NN.sigmoid(z3)
    
    prediction = []
    for i in range(A3.shape[0]):
        max_value = 0
        for j in range(num_labels):
            if A3[i,j] > max_value:
                temp = j
                max_value = A3[i,j]
        prediction.append(temp)
    return prediction

def accuracy(prediction, y):
    correct_count = 0
    for i in range(len(y)):
        if prediction[i] == y[i]:
            correct_count += 1
    acc = correct_count/m*100
    return acc

def plotImage(x):
    imgdata = x.reshape((28, 28))
    plt.imshow(imgdata, cmap = 'gray')

if __name__ == "__main__":
    f = sio.loadmat("MNIST_thetas.mat")
    g = sio.loadmat("MNIST_test.mat")
    Theta1 = f['Theta1']
    Theta2 = f['Theta2']
    X = g['X']
    y = g['y']
    y = y.flatten()
    X = Norm.Normalize(X)
    
    num_labels = 10
    m = len(y)
    x = X[5465].reshape((1, 784))
    
    prediction = predict(X, Theta1, Theta2, m)
    acc = accuracy(prediction, y)
    #print("The predicted value is:", prediction)
    print("Training set accuracy:", acc)
    #plotImage(x)
