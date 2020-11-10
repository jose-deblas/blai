import numpy as np
from blai_functions import *

def blai_layer_sizes(X,Y,hidden_units):
    """
    Argument:
    X -- dataset of shape (number of features, number of examples)
    Y -- labels of shape (number of output values, number of examples)
    hidden_units -- list with #hidden units per layer

    Returns:
    layers -- List with the size of each layer, including input (layers[0] and output (layer[L])
    """
    layers = hidden_units
    layers.insert(0,X.shape[0])
    layers.insert(len(layers),Y.shape[0])

    return layers

def blai_initialize_parameters(units_per_layer, blai_seed):
    """
    Argument:
    units_per_layer -- vector with size of the layers
    blai_seed -- if we want the same results set a seed
    
    Returns:
    params -- python dictionary containing your parameters:
                    from
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    to
                    WL -- weight matrix of shape (n_y, n_h_L)
                    bL -- bias vector of shape (n_y, 1)
    """
    
    # we set up a seed so that your output matches ours although the initialization is random.
    #np.random.seed(2) 
    if(blai_seed > 0):
        np.random.seed(blai_seed)
    
    stabilizer = 0.01
    parameters = {}
    L = len(units_per_layer)

    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(units_per_layer[l], units_per_layer[l-1])*stabilizer
        parameters["b" + str(l)] = np.zeros((units_per_layer[l], 1))

    return parameters


def blai_forward_propagation(X, parameters, act_func, L_act_func):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    act_func -- activation function for layers from l=1 to l=L-1
    L_act_func -- activation function for the last layer L
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """

    L = len(parameters) // 2
    cache = {}

    for l in range(1,L):
        if(l == 1):
            A = X
        else:
            A = cache["A"+str(l-1)]

        #linear func
        cache["Z"+str(l)] = np.dot(parameters["W"+str(l)], A) + parameters["b"+str(l)]

        #activation func
        cache["A"+str(l)] = blai_g(cache["Z"+str(l)], act_func)

    #Notice that the loop arrives until L-1, So we compute the last layer L with L_act_func
    cache["Z"+str(L)] = np.dot(parameters["W"+str(L)],cache["A"+str(L-1)])+parameters["b"+str(L)]
    cache["A"+str(L)] = blai_g(cache["Z"+str(L)], L_act_func)

    assert(cache["A"+str(L)].shape == (1, X.shape[1]))
            
    return cache["A"+str(L)], cache

def blai_compute_cost(AL, Y):
    """
    Computes the cross-entropy cost given in equation for gradient descent
    𝐽=−1/m ∑𝑖 (y(𝑖)log(a[L](𝑖))+(1−y(𝑖))log(1−a[L](𝑖)))
    
    Arguments:
    AL -- The g(z) function value of last layer, the output or Yhat. Shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    cost -- cross-entropy cost given equation for gradient descent and sigmoid
    
    """
    
    # number of examples
    m = Y.shape[1]

    # Compute the cross-entropy cost
    logprobs = np.multiply(Y,np.log(AL)) + np.multiply((1-Y),np.log(1-AL))
    cost = - np.sum(logprobs)/m
    
    # makes sure cost is the dimension we expect. Turns [[17]] into 17 
    cost = float(np.squeeze(cost))
    assert(isinstance(cost, float))
    
    return cost

def blai_backward_propagation(parameters, cache, X, Y, act_func):
    """
    Implement the backward propagation.
    
    Arguments:
    parameters -- python dictionary containing our parameters W[i] and b[i]
    cache -- a dictionary containing Z[i] and A, "Z2" and "A2".
    X -- input data of shape (n_x, m)
    Y -- "true" labels vector of shape (1, m)
    act_func -- activation function for layers from l=1 to l=L-1
    L_act_func -- activation function for the last layer L
   
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    L = len(parameters) // 2
    grads = {}

    #The next reversed loop will start in L-1. So we calculate the gradients on L layer
    grads["dZ"+str(L)] = cache["A"+str(L)] - Y
    grads["dW"+str(L)] = np.dot(grads["dZ"+str(L)], cache["A"+str(L-1)].T) / m
    grads["db"+str(L)] = np.sum(grads["dZ"+str(L)], axis = 1, keepdims=True) / m

    #And now the rest of
    for l in reversed(range(1,L)):

        grads["dZ"+str(l)] = np.multiply(parameters["W"+str(l+1)].T, grads["dZ"+str(l+1)]) * blai_derivative_g(cache, l, act_func)

        if(l == 1):
            A = X.T
        else:
            A = cache["A"+str(l-1)].T

        grads["dW"+str(l)] = np.dot(grads["dZ"+str(l)], A) / m
        grads["db"+str(l)] = np.sum(grads["dZ"+str(l)], axis = 1, keepdims=True) / m
        

    return grads

def blai_update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Arguments:
    parameters -- python dictionary containing our parameters W[i] and b[i]
    grads -- dictionary containing all the derivatives
    learning rate -- hyperparameter to increase or decrease the learning
    
    Returns:
    parameters -- parameters learnt by the model: params = params - (learning_rate * grads)
    """
    L = len(parameters) // 2

    for l in range(1,L+1):
        parameters["W"+str(l)] = parameters["W"+str(l)] - (learning_rate * grads["dW"+str(l)])
        parameters["b"+str(l)] = parameters["b"+str(l)] - (learning_rate * grads["db"+str(l)])

    return parameters

def blai_nn_model(X, Y, units_hidden_layers, act_func="relu", L_act_func="relu", learning_rate = 1.2, num_iterations = 1000, print_cost=0, blai_seed=0):
    """
    Arguments:
    X -- dataset of shape (number of features, number of examples)
    Y -- labels of shape (number of output values, number of examples)
    units_hidden_layers -- vector with size of each hidden_layer, excluding input and output layers
    act_func -- activation function for layers from l=1 to l=L-1. ReLU by default
    L_act_func -- activation function for the last layer L. ReLU by default
    learning_rate -- hyperparameter alpha to update parameters
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if print_cost > 0 the cost is printed every print_cost iterations
    blai_seed -- to set a seed along blai functions

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    #np.random.seed(3)
    if(blai_seed > 0):
        np.random.seed(blai_seed)

    units_per_layer = blai_layer_sizes(X, Y, units_hidden_layers)
    
    parameters = blai_initialize_parameters(units_per_layer, blai_seed)

    for i in range(0, num_iterations):

        # Forward propagation
        AL, cache = blai_forward_propagation(X, parameters, act_func, L_act_func)
        
        # Cost function
        cost = blai_compute_cost(AL, Y)
 
        # Backpropagation
        grads = blai_backward_propagation(parameters, cache, X, Y, act_func)
 
        # Gradient descent parameter update
        parameters = blai_update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every i iterations
        if print_cost > 0 and i % print_cost == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def blai_predict(parameters, X, act_func="tanh", L_act_func="sigmoid"):
    """
    Using the learned parameters, predicts Y for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    #binary
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = blai_forward_propagation(X, parameters=parameters, act_func=act_func, L_act_func=L_act_func)

    predictions = A2 > 0.5
    
    return predictions