import pickle
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp = np.sum(exp_a)
    y = exp_a / sum_exp

    return y


# nomalize True일시 0~255 범위의 픽셀의 값을 0.0~1.0범위로 변환
# 데이터를 특정 범위로 변환하는 처리를 정규화
# 신경망의 입력 데이터에 특정 변환을 가하는 것을 전처리
def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# pickle 파일인 sample_weight.pkl에 저장된 학습된 가중치 매개변수를 읽음
# 해당 파일에는 가중치와 편향 매개변수가 딕셔너리 변수로 저장되어 있음
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)

    return y

# x, _ = get_data()
# network = init_network()
# W1, W2, W3 = network['W1'], network['W2'], network['W3']
# print(x.shape)
# print(x[0].shape)
# print(W1.shape)
# print(W2.shape)
# print(W3.shape)

# 하나로 묶은 입력 데이터를 배치
# 배치처리가 유리한 이유는 수치 계산 라이브러리 대부분이 큰 배열을 효율적으로 처리
# 느린 I/O를 통해 데이터를 읽는 횟수가 줄어, 빠른 CPU나 GPU로 순수 계산을 수행하는 비율이 높아짐
# 즉, 배치처리를 수행함으로써 큰 배열로 이뤄진 계산을 하게 되는데,
# 컴퓨터에서는 큰 배열을 한꺼번에 계산하는 것이 분할된 작은 배열을 여러번 계산하는 것보다 빠름

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0,len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

x= np.array([[0.1,0.8,0.1],[0.3,0.1,0.6],[0.2,0.5,0.3],[0.8,0.1,0.1]])
y = np.argmax(x,axis=1) #axis=0 행, axis=1 열
print(y)
# 0.8, 0.6, 0.5, 0.8 이 각각 큰 값이므로 인덱스로 표현하면 1, 2, 1, 0 의 값이 나온다

y = np.array([1,2,1,0])
t = np.array([1,2,0,0])
print(y==t)
print(np.sum(y==t))