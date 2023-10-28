import sys, os

import gradient_simplenet
import two_layer_net

import numpy as np

# net = gradient_simplenet.simpleNet()
# print("net param : ", net.W) #가중치 매개변수
#
# x =np.array([0.6,0.9])
# p = net.predict(x)
# print("p is : ", p)
#
# print("np.argmax(p) : ", np.argmax(p))
# # 최대값의 인덱스
#
# t = np.array([0,0,1]) # 정답 레이블이라 가정
# net.loss(x,t) #loss 값 계산
#
# def f(W):
#     return net.loss(x,t)
#
# # f = lambda W : net.loss(x,t)
#
# dW = gradient_simplenet.numerical_gradient(f, net.W)
# print(dW)

net = two_layer_net.TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
# print(net.params['W1'].shape)
# print(net.params['b1'].shape)
# print(net.params['W2'].shape)
# print(net.params['b2'].shape)
#
# x = np.random.rand(100,784) #더미 입력 데이터 100장 분량
# y = net.predict(x)
# print(y.shape)
# print(y[0])

x = np.random.rand(100,784)
t = np.random.rand(100,10)

grads = net.numerical_gradient(x,t) #기울기 계산

print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)