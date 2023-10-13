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

x, t = get_data()
network = init_network()

accuracy_cnt = 0

for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻음 즉, 예측한 값
    if p == t[i]: # 예측한 값과 정답 레이블과 비교하여 정답 카운팅
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x))) # 93%


