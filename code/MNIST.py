import numpy as np
import pandas as pd
import NN_Backprop_fns as NN
import scipy.optimize as opt
import scipy.io as sio
import MNIST_normalize as Norm

f = open("mnist_train.csv", "r")
data = pd.read_csv(f, header = None)
data = np.array(data)
X = data[:, 1:data.shape[1]]
y = data[:, 0]
X = Norm.Normalize(X)

input_layer_size = 784
hidden_layer_size = 25
num_labels = 10
lamb = 1
epsilon1 = 6**0.5/(input_layer_size + hidden_layer_size)**0.5
epsilon2 = 6**0.5/(hidden_layer_size + num_labels)**0.5
                  
Theta1_initial = np.random.rand(hidden_layer_size, input_layer_size+1)*2*epsilon1 - epsilon1
Theta2_initial = np.random.rand(num_labels, hidden_layer_size+1)*2*epsilon2 - epsilon2
nn_params_initial = np.concatenate((Theta1_initial.flatten(), Theta2_initial.flatten()))
nItr = 100              # no. of iterations

nn_params = opt.fmin_cg(NN.Cost, nn_params_initial, fprime = NN.Gradient, args=(input_layer_size, hidden_layer_size, num_labels, X, y, lamb), maxiter = nItr)
Theta1 = nn_params[0:((input_layer_size+1)*hidden_layer_size)].reshape((hidden_layer_size, input_layer_size + 1))
Theta2 = nn_params[(input_layer_size+1)*hidden_layer_size:np.size(nn_params)].reshape((num_labels, hidden_layer_size + 1))

g = {'Theta1':Theta1, 'Theta2':Theta2}
sio.savemat("MNIST_thetas.mat", g)