import sys, os

import gradient_simplenet
import two_layer_net

import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads): # 가중치 매개변수와 기울기
        for key in params.key():
            params[key] -= self.lr * grads[key]

# SGD(확률적 경사 하강법)의 단점
# 지그재그로 (0,0)까지 지그재그로 이동하니 비효율적
# 비등방성(anisotropy)함수 (방향에 따라 성질, 즉 여기에서는 기울기가 달라지는 함수)
# 에서는 탐색 경로가 비효율적이라는 것
# SGD가 지그재그로 탐색하는 근본 원인은 기울어진 방향이 본래의 최솟값과 다른 방향을 가리켜서라는 점도 생각

# SGD의 이러한 단점을 개선해주는 모멘텀, AdaGrad, Adam