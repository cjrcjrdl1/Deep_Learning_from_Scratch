import numpy as np

# softmax() 함수는 컴퓨터로 계산할 때 오버플로 문제 발생
# softmax 함수는 지수함수 사용하는데 지수함수는 쉽게 아주 큰 값 내뱉음
# e^10은 20,000 넘고 e^100은 0만 40개가 넘어가며 e^1000은 무한대인 inf 출력
# 이런 큰 값끼리 나누면 결과 수치가 불안정해짐

# 해결법
# C라는 임의 정수를 분모 분자에 곱하고 C를 exp 안으로 옮겨 logC로 만듦
# 마지막으로 logC를 C`라는 새로운 기호로 바꿈

# a = np.array([1010, 1000, 990])
# print(np.exp(a) / np.sum(np.exp(a))) # 제대로 출력 X

# c = np.max(a)
# print(a-c)
#
# print(np.exp(a-c) / np.sum(np.exp(a-c)))

# 즉 오버플로를 대응해야 제대로 출력된다. 다시 함수를 구현하면

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp = np.sum(exp_a)
    y = exp_a / sum_exp

    return y

a = np.array([0.3,2.9,4.0])
y = softmax(a)
print(y)
print(np.sum(y))

# softmax 함수의 출력은 0에서 1.0 사이의 실수

# softmax를 적용해도 각 원소의 대소 관계는 변하지 않는다.
# 지수 함수 y = exp(x)가 단조 증가 함수이기 때문

# 기계학습에서는 학습과 추론 두단계를 거쳐 이뤄지는데,
# 학습단계에서 모델을 학습하고
# 추론단계에서 학습한 모델로 미지에 데이터에 대해 추론(분류)을 수행
# 추론단계에서는 출력층의 소프트맥스 함수를 생략하는 것이 일반적

# 추론 과정을 신경망의 순전파라고도 함

# MNIST 이미지 데이터는 28x28 크기의 회색조 이미지, 각 픽셀은 0에서 255 값 취함

