import numpy as np
import matplotlib.pyplot as plt

# 인수 x가 실수(부동소수점)만 받아들임
# def step_function(x):
#     if x>0:
#         return 1
#     else:
#         return 0


# def step_function(x):
#     y = x>0
#     return y.astype(np.int32)
#
# x = np.array([-1.0,1.0,2.0])
# print(x)
# y=x>0
# print(y) #계단함수는 0,1중 하나이므로 bool 형태를 int형으로 바꿔주어야 한다.
#
# y=y.astype(np.int32)
# print(y)

# 계단함수
# def step_function(x):
#     return np.array(x>0, dtype=np.int32)
#
# x = np.arange(-5.0,5.0,0.1)
# y = step_function(x)
# plt.plot(x,y)
# plt.ylim(-0.1,1.1)
# plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))

# x= np.array([-1.0,1.0,2.0])
# print(sigmoid(x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()

# 계단 함수와 시그모이드 함수의 공통점은 비선형(직선 1개로는 그릴 수 없는 함수) 함수이다.

# ReLU 함수(렐루 함수)
def relu(x):
    return np.maximum(0,x)
