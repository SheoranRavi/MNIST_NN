'''Certain functions used repeatedly in NNs. Single hidden layer'''
import numpy as np
import scipy.io as sio
def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z)*(1 - sigmoid(z))

def label(y, m, num_labels):            # Change for probs other than digit recognition
    labels = np.zeros((m, num_labels))
    for i in range(m):
        for j in range(num_labels):
            if j == y[i]:               # mapping 0 to 0, 1 to 1 and so on.
                labels[i, j] = 1
    
    '''for i in range(m):
        for j in range(num_labels):     # mapping 0 to 9th output node
            if j == y[i]-1:             # 1 to 0th; 2 to 1st and so on.
                labels[i, j] = 1'''
    return labels
i = 0

def Cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    m = X.shape[0]      # no. of training examples
    Theta1 = nn_params[0:((input_layer_size+1)*hidden_layer_size)].reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[(input_layer_size+1)*hidden_layer_size:np.size(nn_params)].reshape((num_labels, hidden_layer_size + 1))
    J = 0
    X = np.c_[np.ones(m), X]
    z2 = np.dot(X, Theta1.T)
    A2 = sigmoid(z2)
    A2 = np.c_[np.ones(m), A2]
    z3 = np.dot(A2, Theta2.T)
    A3 = sigmoid(z3)
    
    y = label(y, m, num_labels)
    for i in range(m):
        for j in range(num_labels):
            J += y[i,j]*np.log(A3[i, j]) + (1 - y[i, j])*np.log(1-A3[i,j])
    #======= add regularization term to cost=========
    J = -(1/m)*J
    for i in range(hidden_layer_size):
        for j in range(1, input_layer_size+1):
            J += lamb/(2*m)*Theta1[i,j]**2
    for i in range(num_labels):
        for j in range(1, hidden_layer_size+1):
            J += lamb/(2*m)*Theta2[i,j]**2
    return J

def Gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    m = X.shape[0]
    Theta1 = nn_params[0:((input_layer_size+1)*hidden_layer_size)].reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[(input_layer_size+1)*hidden_layer_size:np.size(nn_params)].reshape((num_labels, hidden_layer_size + 1))
    Delta1 = np.zeros(Theta1.shape)
    Delta2 = np.zeros(Theta2.shape)
    X = np.c_[np.ones((m, 1)), X]
    y = label(y, m, num_labels)
    
    for i in range(m):
        a1 = X[i,:]
        a1 = a1.reshape((len(a1), 1))
        z2 = np.dot(Theta1, a1)
        a2 = sigmoid(z2)
        a2 = np.r_[np.ones((1,1)), a2]
        z3 = np.dot(Theta2, a2)
        a3 = sigmoid(z3)
        
        yi = y[i].reshape((num_labels, 1))
        del3 = a3 - yi
        s_grad = sigmoidGradient(z2)
        s_grad = np.r_[np.ones((1,1)), s_grad]
        del2 = np.dot(Theta2.T, del3)*s_grad
        del2 = del2[1:len(del2)]
                     
        Delta1 = Delta1 + np.dot(del2, a1.T)
        Delta2 = Delta2 + np.dot(del3, a2.T)
    D1 = (1/m)*(Delta1 + lamb*Theta1)
    D1[:,0] -= (lamb/m)*Theta1[:,0]
    D2 = (1/m)*(Delta2 + lamb*Theta2)
    D2[:,0] -= (lamb/m)*Theta2[:,0]
    
    # Unroll gradients
    D = np.concatenate((D1.flatten(), D2.flatten()))
    return D
        
if __name__ == "__main__":
    # Test the above functions
    f = sio.loadmat("ex4weights.mat")
    g = sio.loadmat("ex4data1.mat")
    