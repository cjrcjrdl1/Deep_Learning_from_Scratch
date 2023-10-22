import numpy as np
import matplotlib.pylab as plt

def function_2(x):
    return x[0]**2 + x[1]**2

# 모든 변수의 편미분을 벡터로 정리한 것을 기울기(gradient)

# f는 함수 x는 넘파이 배열, 넘파이 배열 x의 각 원소에 대해서 수치 미분을 구함
def numerical_gradient(f,x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성, 그 원소가 모두 0

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원

    return grad

# 경사 하강법
# f는 최적화 함수, init_x는 초기값, lr은 learning rate를 의미하는 학습률
# step_num은 경사법에 따른 반복 횟수
# 함수의 기울기는 numerical_gradient(f,x)로 구하고
# 기울기에 학습률을 곱한 값으로 갱신하는 처리를 step_num번 반복
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

def function_2(x):
    return x[0]**2 + x[1]**2


init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
# 거의 (0,0)에 가까운 값

# 학습률이 너무 큰 예 : lr=10.0, 너무 큰 값으로 발산
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))

# 학습률이 너무 작은 예 : lr=1e-10, 갱신되지 않은 채 끝남
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))

# 학습률 같은 매개변수를 하이퍼파라미터(초매개변수)라고 함
# 이는 가중치와 편향같은 신경망의 매개변수와는 성질이 다른 매개변수
# 신경망의 가중치 매개변수는 훈련 데이터와 학습 알고리즘에 의해서 자동으로 획득되는 매개변수인 반면
# 학습률 같은 하이퍼파라미터는 사람이 직접 설정해야하는 매개변수