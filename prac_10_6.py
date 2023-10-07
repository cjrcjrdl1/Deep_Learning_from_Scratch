# def AND(x1, x2):
#     w1, w2, theta = 0.5,0.5,0.7
#     tmp = x1*w1 + x2*w2
#     if tmp <= theta:
#         return 0
#     elif tmp > theta:
#         return 1
#
# print(AND(0,0))
# print(AND(1,0))
# print(AND(0,1))
# print(AND(1,1))

import numpy as np
# x = np.array([0,1]) # 입력
# w = np.array([0.5,0.5]) # 가중치
# b = -0.7 # 편향(bias)
# print(w*x)
# print(np.sum(w*x))
# print(np.sum(w*x) + b)
#
def AND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b=-0.7
    tmp = np.sum(w*x) + b
    if tmp<=0:
        return 0
    else:
        return 1
#
# print(AND(0,1))
# print(AND(1,1))

def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5, -0.5]) # AND와 가중치 (w와 b)만 다르다
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <=0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.2
    tmp = np.sum(w*x) + b
    if tmp <=0:
        return 0
    else:
        return 1
    
# XOR를 직선 하나로 나눌 방법이 마땅히 생각나지 않음
# 하지만 직선이라는 제약이 없어지면 곡선으로 가능
# 곡선의 영역을 비선형 영역, 직선의 영역을 선형 영역
# 안타깝게도 퍼셉트론으로는 XOR 게이트 표현X
# But, 다층 퍼셉트론으로는 만들 수 있다.
# -> NAND, OR, AND를 통해 구현 가능

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y

print(XOR(0,0))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(1,1))
