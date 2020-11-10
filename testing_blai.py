from blai import *
#import numpy as num_px_x

print("blai_initialize_parameters(units_per_layer, blai_seed)")
units_per_layer = [2,4,1]
blai_seed = 2
parameters = blai_initialize_parameters(units_per_layer, blai_seed)
print("units_per_layer: " + str(units_per_layer))
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

print("\nExpected output:")

print("W1 = [[-0.00416758 -0.00056267]\n"+
"[-0.02136196  0.01640271]\n"+
"[-0.01793436 -0.00841747]\n"+
"[ 0.00502881 -0.01245288]]\n"+
"b1 = [[0.]\n"+
"[0.]\n"+
"[0.]\n"+
"[0.]]\n"+
"W2 = [[-0.01057952 -0.00909008  0.00551454  0.02292208]]\n"+
"b2 = [[0.]]")

print("----------------------------------------------")

def forward_propagation_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    b1 = np.random.randn(4,1)
    b2 = np.array([[ -1.3]])

    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]]),
     'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b1': b1,
     'b2': b2}

    return X_assess, parameters

print("blai_forward_propagation(X, parameters, act_func, L_act_func)")
X_assess, parameters = forward_propagation_test_case()
AL, cache = blai_forward_propagation(X_assess, parameters,"tanh","sigmoid")

#to test we calculalte the mean
print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))
print("\nExpected output:")
print("0.26281864019752443 0.09199904522700109 -1.3076660128732143 0.21287768171914198")

print("----------------------------------------------")
print("blai_compute_cost(AL, Y)")

def compute_cost_test_case():
    np.random.seed(1)
    Y_assess = (np.random.randn(1, 3) > 0)
    a2 = (np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]))
    
    return a2, Y_assess

AL, Y_assess = compute_cost_test_case()
print("cost = " + str(blai_compute_cost(AL, Y_assess)))
print("\nExpected output:")
print("cost = 0.6930587610394646")

print("----------------------------------------------")

print("blai_backward_propagation(parameters, cache, X, Y, act_func)")

def backward_propagation_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = (np.random.randn(1, 3) > 0)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]]),
     'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b1': np.array([[ 0.],
        [ 0.],
        [ 0.],
        [ 0.]]),
     'b2': np.array([[ 0.]])}

    cache = {'A1': np.array([[-0.00616578,  0.0020626 ,  0.00349619],
         [-0.05225116,  0.02725659, -0.02646251],
         [-0.02009721,  0.0036869 ,  0.02883756],
         [ 0.02152675, -0.01385234,  0.02599885]]),
  'A2': np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]),
  'Z1': np.array([[-0.00616586,  0.0020626 ,  0.0034962 ],
         [-0.05229879,  0.02726335, -0.02646869],
         [-0.02009991,  0.00368692,  0.02884556],
         [ 0.02153007, -0.01385322,  0.02600471]]),
  'Z2': np.array([[ 0.00092281, -0.00056678,  0.00095853]])}
    return parameters, cache, X_assess, Y_assess

parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = blai_backward_propagation(parameters, cache, X_assess, Y_assess, "tanh")
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))

print("\nExpected output:")
print("**dW1**	[[ 0.00301023 -0.00747267] [ 0.00257968 -0.00641288] [-0.00156892 0.003893 ] [-0.00652037 0.01618243]]")
print("**db1**	[[ 0.00176201] [ 0.00150995] [-0.00091736] [-0.00381422]]")
print("**dW2**	[[ 0.00078841 0.01765429 -0.00084166 -0.01022527]]")
print("**db2**	[[-0.16655712]]")
print("----------------------------------------------")

print("blai_update_parameters(parameters, grads, learning_rate = 1.2):")

def update_parameters_test_case():
    parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],
        [-0.02311792,  0.03137121],
        [-0.0169217 , -0.01752545],
        [ 0.00935436, -0.05018221]]),
 'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
 'b1': np.array([[ -8.97523455e-07],
        [  8.15562092e-06],
        [  6.04810633e-07],
        [ -2.54560700e-06]]),
 'b2': np.array([[  9.14954378e-05]])}

    grads = {'dW1': np.array([[ 0.00023322, -0.00205423],
        [ 0.00082222, -0.00700776],
        [-0.00031831,  0.0028636 ],
        [-0.00092857,  0.00809933]]),
 'dW2': np.array([[ -1.75740039e-05,   3.70231337e-03,  -1.25683095e-03,
          -2.55715317e-03]]),
 'db1': np.array([[  1.05570087e-07],
        [ -3.81814487e-06],
        [ -1.90155145e-07],
        [  5.46467802e-07]]),
 'db2': np.array([[ -1.08923140e-05]])}
    return parameters, grads

parameters, grads = update_parameters_test_case()

parameters = blai_update_parameters(parameters, grads, learning_rate = 1.2)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print("\nExpected output:")
print("**W1**	[[-0.00643025 0.01936718] [-0.02410458 0.03978052] [-0.01653973 -0.02096177] [ 0.01046864 -0.05990141]]")
print("**b1**	[[ -1.02420756e-06] [ 1.27373948e-05] [ 8.32996807e-07] [ -3.20136836e-06]]")
print("**W2**	[[-0.01041081 -0.04463285 0.01758031 0.04747113]]")
print("**b2**	[[ 0.00010457]]")

print("----------------------------------------------")

print("blai_nn_model(X, Y, units_per_layer, act_func=relu, L_act_func=relu, learning_rate=1.2, num_iterations=1000, print_cost=0, blai_seed=0)")

def nn_model_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = (np.random.randn(1, 3) > 0)
    
    return X_assess, Y_assess

X_assess, Y_assess = nn_model_test_case()
parameters = blai_nn_model(X_assess, Y_assess, [4], act_func="tanh", L_act_func="sigmoid", num_iterations=10000, print_cost=3000, blai_seed = 3)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

print("\nExpected output:")
print("**cost after iteration 0**	0.693198")
print("**W1**	[[ 0.56305445 -1.03925886],[ 0.7345426  -1.36286875],[-0.72533346  1.33753027],[ 0.74757629 -1.38274074]]")
print("**b1**	[[-0.22240654],[-0.34662093],[ 0.33663708],[-0.35296113]]")
print("**W2**	[[ 1.82196893  3.09657075 -2.98193564  3.19946508]]")
print("**b2**	[[ 0.21344644]]")

print("----------------------------------------------")

print("def blai_predict(parameters, X, type = 'binary')")

def predict_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],
        [-0.02311792,  0.03137121],
        [-0.0169217 , -0.01752545],
        [ 0.00935436, -0.05018221]]),
     'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
     'b1': np.array([[ -8.97523455e-07],
        [  8.15562092e-06],
        [  6.04810633e-07],
        [ -2.54560700e-06]]),
     'b2': np.array([[  9.14954378e-05]])}
    return parameters, X_assess

parameters, X_assess = predict_test_case()

predictions = blai_predict(parameters, X_assess, act_func="tanh", L_act_func="sigmoid")
print("predictions mean = " + str(round(np.mean(predictions),12)))

print("\nExpected output:")
print("**predictions mean**	0.666666666667")

print("----------------------------------------------")
