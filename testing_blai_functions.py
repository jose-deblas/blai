from blai_functions import *

def blai_array_compare(array1, array2):

    
    value1 = np.sum(array1)
    value2 = np.sum(array2)
    total = round(value1,5)-round(value2,5)

    print(value1)
    print(value2)
    print(total)

    return total==0.0

num_errors = 0

print("----------------------------------------------")

print("blai_relu(x)")
x = np.array([[0.731,-0.891, 0.952],[-0.123,0.566,-0.263]])
expected_output = np.array([[0.731,0.,0.952],[0.,0.566,0.]])
if (blai_array_compare(expected_output, blai_relu(x))):
    print("...ok")
else:
    print ("ERROR: value not expected")
    print(blai_relu(x))
    print("Expected Output:")
    print(expected_output)
    num_errors+=1

print("----------------------------------------------")

print("blai_sigmoid(x)")
x = np.array([[1, 2, 3],[1, 2, 3]])
expected_output = np.array([[0.73105858, 0.88079708, 0.952574135],[0.73105858, 0.88079708, 0.952574135]])
s = blai_sigmoid(x)

if (blai_array_compare(expected_output, s)):
    print("...ok")
else:
    print ("ERROR: value not expected")
    print(s)
    print("Expected Output:")
    print(expected_output)
    num_errors+=1

print("----------------------------------------------")

print("blai_tanh(x)")
x = np.array([[ 1.7386459, 1.74687437, 1.74830797],[-0.81350569, -0.73394355, -0.78767559]])
expected_output = np.array([[ 0.9400694, 0.94101876, 0.94118266],[-0.67151964, -0.62547205, -0.65709025]])
a = blai_tanh(x)

if (blai_array_compare(expected_output, a)):
    print("...ok")
else:
    print ("ERROR: value not expected")
    print(a)
    print("Expected Output:")
    print(expected_output)
    num_errors+=1

print("----------------------------------------------")

print("blai_softmax(x)")
x = np.array([[9, 2, 5, 0, 0],[7, 5, 0, 0 ,0]])
a = blai_softmax(x)
expected_output = np.array([[9.80897665e-01, 8.94462891e-04, 1.79657674e-02, 1.21052389e-04, 1.21052389e-04],
    [8.78679856e-01, 1.18916387e-01, 8.01252314e-04, 8.01252314e-04, 8.01252314e-04]])

if (blai_array_compare(expected_output, a)):
    print("...ok")
else:
    print ("ERROR: value not expected")
    print(a)
    print("Expected Output:")
    print(expected_output)
    num_errors+=1

print("----------------------------------------------")

print("blai_relu_derivative(x)")
#We need to know dZ = dA*relu_derivative(Z)
dA = np.array([[-0.41675785, -0.05626683]])
Z = np.array([[0.04153939, -1.11792545]])
dZ = dA*blai_relu_derivative(Z)
expected_output = np.array([[-0.41675785, 0.]])

if (blai_array_compare(expected_output, dZ)):
    print("...ok")
else:
    print ("ERROR: value not expected")
    print(dZ)
    print("Expected Output:")
    print(expected_output)
    num_errors+=1

print("----------------------------------------------")

print("sigmoid_derivative(x)")
x = np.array([1, 2, 3])
derivative_sigmoid = blai_sigmoid_derivative(blai_sigmoid(x))
expected_output = np.array([0.19661193, 0.10499359, 0.04517666])

if (blai_array_compare(expected_output, derivative_sigmoid)):
    print("...ok")
else:
    print ("ERROR: value not expected")
    print(derivative_sigmoid)
    print("Expected Output:")
    print(expected_output)
    num_errors+=1

print("----------------------------------------------")

print("blai_tanh_derivative(A)")
A = np.array([[-0.00616578, 0.0020626, 0.00349619],[-0.05225116, 0.02725659, -0.02646251],
    [-0.02009721, 0.0036869, 0.02883756],
    [0.02152675, -0.01385234, 0.02599885]])
d_tanh = blai_tanh_derivative(A)
expected_output = np.array([[0.99996198,0.99999575,0.99998778],[0.99726982,0.99925708,0.99929974],
    [0.9995961,0.99998641,0.9991684],
    [0.9995366,0.99980811,0.99932406]])

if (blai_array_compare(expected_output, d_tanh)):
    print("...ok")
else:
    print ("ERROR: value not expected")
    print(d_tanh)
    print("Expected Output:")
    print(expected_output)
    num_errors+=1

print("----------------------------------------------")

print ("image2vector(image)")
# This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],
       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]]])

image_vector = blai_image2vector(image)
expected_output = np.array([[0.67826139],[0.29380381],[0.90714982],[0.52835647],[0.4215251],[0.45017551],[0.92814219],[0.96677647],[0.85304703],[0.52351845],[0.19981397],[0.27417313]])

if (blai_array_compare(expected_output, image_vector)):
    print("...ok")
else:
    print ("ERROR: value not expected")
    print(image_vector)
    print("Expected Output:")
    print(expected_output)
    num_errors+=1

print("----------------------------------------------")

print ("normalizeRows(x)")
x = np.array([[0, 3, 4],[1, 6, 4]])
xnorm = blai_normalizeRows(x)
expected_output = np.array([[0.,0.6,0.8],[0.13736056,0.82416338,0.54944226]])

if (blai_array_compare(expected_output, xnorm)):
    print("...ok")
else:
    print ("ERROR: value not expected")
    print(xnorm)
    print("Expected Output:")
    print(expected_output)
    num_errors+=1

print("----------------------------------------------")

print("L1 loss function")
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
loss_1 = blai_loss_1(yhat,y)
expected_output = 1.1

if (loss_1 == expected_output):
    print("...ok")
else:
    print ("ERROR: value not expected")
    print(loss_1)
    print("Expected Output:")
    print(expected_output)
    num_errors+=1

print("----------------------------------------------")

print("L2 loss function")
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
loss_2 = blai_loss_2(yhat,y)
expected_output = 0.43

if (loss_2 == expected_output):
    print("...ok")
else:
    print ("ERROR: value not expected")
    print(loss_2)
    print("Expected Output:")
    print(expected_output)
    num_errors+=1

print("----------------------------------------------")
print("----------------------------------------------")
print("Total Errors: " + str(num_errors))
print("----------------------------------------------")
print("----------------------------------------------")
