# coding: utf-8

import numpy as np

def blai_g(Z, g):
    """
    Compute g(Z)
    Remember that g is the activation function [sigmoid|tanh|relu|softmax]

    Arguments:
    Z -- The result of the linear function
    g -- The activation function to apply

    Return:
    result -- result of the g(Z)
    """
    activation = {
        "relu": blai_relu(Z),
        "sigmoid": blai_sigmoid(Z),
        "tanh": blai_tanh(Z),
        "softmax": blai_softmax(Z)
    }

    return activation.get(g, "activation function g " + g + " not implemented")

def blai_relu(Z):
    """
    Compute the relu of Z using np.maximum(0,Z)
    Remember that Z values less than zero are set to zero

    Arguments:
    Z -- A scalar or numpy array of any size

    Return:
    r -- relu(Z)
    """
    
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)

    return A

def blai_sigmoid(Z):
    """
    Compute the sigmoid of Z
    Remember that sigmoid(Z) == 1 / (1+e**-Z) 
    where e is Euler number AKA Napier constant = 2.71828

    Arguments:
    Z -- A scalar or numpy array of any size

    Return:
    A -- sigmoid(Z)
    """
    
    A = 1 / (1 + np.exp(-Z))
    
    assert(A.shape == Z.shape)
    
    return A

def blai_tanh(Z):
    """
    Compute the tanh of Z
    Remember that tanh(Z) == (e**Z - e**-Z) / (e**Z + e**-Z)

    Arguments:
    Z -- A scalar or numpy array of any size

    Return:
    A -- tanh(Z)
    """
    
    #the easiest way
    #A = np.tanh(Z)
    e_exp_Z = np.exp(Z)
    e_exp_minusZ = np.exp(-Z)
    A = (e_exp_Z - e_exp_minusZ) / (e_exp_Z + e_exp_minusZ)

    assert(A.shape == Z.shape)
    
    return A

def blai_softmax(Z):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (m,n).

    Argument:
    Z -- A numpy matrix of shape (m,n)

    Returns:
    A -- A numpy matrix equal to the softmax of x, of shape (m,n)
    """
    
    # Apply exp() element-wise to Z
    Z_exp = np.exp(Z)

    # Create a vector X_sum that sums each row of Z_exp using np.sum(..., axis = 1, keepdims = True)
    X_sum = np.sum(Z_exp, axis=1, keepdims=True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    A = Z_exp / X_sum
    
    return A

def blai_derivative_g(cache, layer, g):
    """
    Compute g'(Z)
    Remember that g'(Z) is the derivative of the activation function g(Z) [sigmoid|tanh|relu|softmax]

    Arguments:
    cache -- cache stores A and Z to make an efficient execution along different detivatives of g
    layer -- the layer to select A or Z values from cache
    g -- The activation function that was applied with Z

    Return:
    result -- result of the g'(Z)
    """
    derivative = {
        "relu": blai_relu_derivative(cache["Z"+str(layer)]),
        "sigmoid": blai_sigmoid_derivative(cache["A"+str(layer)]),
        "tanh": blai_tanh_derivative(cache["A"+str(layer)])
        #"softmax": blai_softmax_derivative(A)
    }

    return derivative.get(g, "derivative function g' " + g + " not implemented")

def blai_relu_derivative(Z):
    """
    Compute the derivative of relu(Z)
    Remember that derivative relu(Z) = (Z >= 0)

    Arguments:
    Z -- A scalar or numpy array of any size

    Return:
    r -- relu(Z)
    """
    #Z[Z <= 0] = 0

    #return Z
    return Z >= 0


def blai_sigmoid_derivative(A):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    Remember that d𝜎(𝑥)=𝜎*(1−𝜎)
    Arguments:
    A -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """

    dA = A*(1-A)
    
    return dA

def blai_tanh_derivative(A):
    """
    Compute the derivative of tanh(Z)
    Remember that derivative tanh(Z) = 1 - A**2

    Arguments:
    Z -- A scalar or numpy array of any size

    Return:
    r -- derivative of tanh(Z) = 1 - A**2
    """
    
    return 1 - A**2


def blai_image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    v = np.reshape(image,(image.shape[0]*image.shape[1]*image.shape[2],1))
    ### END CODE HERE ###
    
    return v



def blai_normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    Remember: To normalize a vector, divide each row vector of x by its norm. 
    The norm of row vector is ||x|| = sqrt(x1**2 + x2**2 + ... + xi**2)
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    #is worse but step by step
    #xnorm = np.sum(x**2,axis=1)
    #xnorm = np.sqrt(xnorm)
    #xnorm = xnorm.reshape(xnorm.shape[0],1)
    
    # Divide x by its norm.
    x = x/x_norm
    ### END CODE HERE ###

    return x




#Loss function L1
# $$\begin{align*} & L_1(\hat{y}, y) = \sum_{i=0}^m|y^{(i)} - \hat{y}^{(i)}| \end{align*}\tag{6}$$
def blai_loss_1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum(abs(y-yhat))
    ### END CODE HERE ###
    
    return loss


#Loss function L2 
#L2 loss is defined as $$\begin{align*} & L_2(\hat{y},y) = \sum_{i=0}^m(y^{(i)} - \hat{y}^{(i)})^2 \end{align*}\tag{7}$$
def blai_loss_2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    yminusyhat = y - yhat
    loss = np.sum(np.dot(yminusyhat,yminusyhat))
    ### END CODE HERE ###
    
    return loss

