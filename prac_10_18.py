import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist


# y가 1차원이라면, 즉 데이터 하나당 교차 엔트로피 오차를 구하는 경우
# reshape 함수로 데이터의 형상을 바꿔줌
# 배치의 크기로 나눠 정규화하고 이미지 1장당 평균의 교차 엔트로피 오차를 계싼
def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1, y.size)

        batch_size = y.shape[0]
        return -np.sum(t*np.log(y)) / batch_size

# 원-핫 인코딩이 아닌 2나 7 등의 숫자 레이블로 주어질 경우
# 이 구현에선 원 핫 인코딩일 때 t가 0인 원소는 교차 엔트로피 오차도 0이고 그 계산은 무시해도 좋다는 것이 핵심
# 정답에 해당하는 신경망의 출력만으로 교차 엔트로피 오차를 계산할 수 있다.
# 그래서 원 핫 인코딩시 t*np.log(y)였떤 부분을
# 레이블 표현시 np.log(y[np.arange(batch_size),t])) / batch_size 로 구현
# 0~ batch_size-1 까지 배열 생성
def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1, y.size)

        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

# 왜 손실함수를 사용하는가?
# 궁극적인 목적은 높은 정확도를 끌어내는 매개변수 값을 찾는 것
# 그렇다면 정확도라는 지표를 놔두고 손실 함수의 값이라는 우회적인 방법을 택하는 이유는?

# 신경망 학습에서는 최적의 매개변수(가중치와 편향)을 탐색할 때 손실 함수의 값이 가능한 한 작게 하는 매개변수 값을 찾는다.
# 이때 매개변수의 미분(기울기)을 계산하고, 그 미분 값을 단서로 매개변수의 값을 서서히 갱신하는 과정 반복

# 신경망을 학습할 때 정확도를 지표로 삼아서는 안된다.
# 정확도를 지표로 하면 매개변수의 미분이 대부분의 장소에서 0이 되기 때문이다.(0이 되면 갱신 멈춤)
