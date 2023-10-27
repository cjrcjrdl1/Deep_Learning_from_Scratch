import numpy as np
import matplotlib.pylab as plt

def function_2(x):
    return x[0]**2 + x[1]**2

def function_1(x):
    return x[0] **2 + x[1] **2

# X = np.arange(-10,10,0.1)
# Y = np.arange(-10,10,0.1)
# X, Y = np.meshgrid(X,Y)
#
# Z = function_1(np.array([X,Y]))
#
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# surf = ax.plot_wireframe(X, Y, Z , alpha = 0.5)
# plt.xlabel("x0")
# plt.ylabel("x1")
# plt.show()

# def numerical_diff(f,x):
#     h = 1e-4
#     return (f(x+h) - f(x-h)) / (2*h)
# # x1을 상수로 간주
# def function_tmp1(x0):
#     return x0*x0 + 4.0**2.0
# # x0을 상수로 간주
# def function_tmp2(x1):
#     return 3.0**2.0 + x1*x1
#
# print(numerical_diff(function_tmp1,3.0))
# print(numerical_diff(function_tmp2,4.0))

def numerical_gradient(f,x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

print(numerical_gradient(function_2, np.array([3.,4.])))
print(numerical_gradient(function_2, np.array([0.,2.])))
print(numerical_gradient(function_2, np.array([3.,0.])))

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr*grad
    return x

def function_2(x):
    return x[0] ** 2 + x[1] ** 2

init_x = np.array([-3.,4.])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))

