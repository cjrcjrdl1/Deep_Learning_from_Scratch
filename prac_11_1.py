import sys, os

import gradient_simplenet
import two_layer_net

import numpy as np

X = np.random.rand(2) # 입력
W = np.random.rand(2,3) # 가중치
B = np.random.rand(3) # 편향

print(X.shape)
print(W.shape)
print(B.shape)

Y = np.dot(X,W) + B

# 신경망의 순전파 때 수행하는 행렬의 내적은 기하학에서는 어파인 변환이라고 한다.

X_dot_W = np.array([[0,0,0],[10,10,10]])
B = np.array([1,2,3])

print(X_dot_W)
print(X_dot_W+B)

dY = np.array([[1,2,3],[4,5,6]])
print(dY)

dB = np.sum(dY, axis=0)
print(dB)