import numpy as np
import matplotlib.pyplot as plt

# A = np.array([1,2,3,4])
# print(A)
# print(np.ndim(A)) # 배열의 차원수
# print(A.shape)
# print(A.shape[0])
#
# B = np.array([[1,2],[3,4],[5,6]])
# print(B)
# np.ndim(B)
# print(B.shape) # 3행 2열이라 (3,2) 튜플로 반환

# 행렬의 내적은 가로 곱하기 세로 즉 행 곱하기 열 형식

# A = np.array([[1,2],[3,4]])
# print(A.shape)
# B = np.array([[5,6],[7,8]])
# print(B.shape)
# print(np.dot(A,B)) # 내적

# A = np.array([[1,2,3], [4,5,6]])
# print(A.shape)
# B = np.array([[1,2],[3,4],[5,6]])
# print(B.shape)
# print(np.dot(A,B))

# A = np.array([[1,2],[3,4],[5,6]])
# print(A.shape)
# B = np.array([7,8])
# print(B.shape)
# print(np.dot(A,B))

# X = np.array([1,2])
# print(X.shape)
# W = np.array([[1,3,5],[2,4,6]])
# print(W)
# print(W.shape)
# Y = np.dot(X,W)
# print(Y)

X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X,W1) + B1

def sigmoid(x):
    return 1/(1+np.exp(-x))

Z1 = sigmoid(A1)
print(A1)
print(Z1)

W2 = np.array([[0.1, 0.4], [0.2,0.5], [0.3,0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

def identity_function(x):
    return x

W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

# 구현 정리
def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3], [0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([0.1,0.5])
y = forward(network, x)
print(y)

# 신경망의 순방향 구현
# 신경망은 분류와 회귀 모두에 이용할 수 있다.
# 일반적으로 회귀에는 항등함수를 분류에는 소프트맥스 함수를 사용
# 분류는 사진 속 인물의 성별 분류
# 회귀는 사진 속 인물의 몸무게 예측
