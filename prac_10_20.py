import numpy as np
import matplotlib.pylab as plt

def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1) # 0에서 20까지 0.1 간격의 배열 x를 만든다.
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()

def numerical_diff(f,x):
    h=1e-4 # 0.0001
    return (f(x+h) - f(x-h))/(2*h)

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))

def function_2(x):
    return x[0]**2 + x[1]**2
#     또는 return np.sum(x**2)

# 위처럼 변수가 2개인 경우 어떤 변수에 대한 미분이냐 구별해야함
# 변수가 여럿인 함수에 대한 미분을 편미분

# x0 = 3, x1 = 4 일때 x0에 대한 편미분
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

print(numerical_diff(function_tmp1, 3.0))

# x0 = 3, x1 = 4 일때 x1에 대한 편미분
def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

print(numerical_diff(function_tmp2, 4.0))

# 편미분까지 학습 완