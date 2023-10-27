import numpy as np
import matplotlib.pylab as plt

# h를 매우 작은 수로 입력 : 종종 반올림 오차 발생
print(np.float32(1e-50))
print(np.float64(1e-50))

# 전향차분법보다 중앙차분법이 훨씬 오차를 적게 리턴한다.
def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*(x**2) + 0.1*x

# x = np.arange(0.0,20.0,0.1)
# y = function_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x,y)
# plt.show()

# x=5 일때 기울기
print(numerical_diff(function_1, 5))
# x=10 일때 기울기
print(numerical_diff(function_1, 10))

def tangent_line(f,x):
    d = numerical_diff(f,x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

x = np.arange(0.0,20.0,0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x,y)
plt.plot(x,y2)
plt.show()