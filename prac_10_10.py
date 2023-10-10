# 항등함수는 입력을 그대로 출력.
# 입력과 출력이 항상 같다는 뜻의 항등
# 소프트맥스 함수 exp(x)는 e^x을 뜻하는 지수함수 n은 출력층의 뉴런 수, yk는 그중 k번째 출력
import numpy as np

a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a) # 지수 함수
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
