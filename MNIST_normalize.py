import pandas as pd
import numpy as np

def Normalize(X):
    X = X.astype(np.float64)
    a = np.max(X, axis = 1)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = X[i, j]/a[i]
    return X

if __name__ == '__main__':
    f = open("mnist_train.csv", "r")
    data = pd.read_csv(f, header = None)
    data = np.array(data)
    X = data[:, 1:data.shape[1]]
    y = data[:, 0]
    X = Normalize(X)
