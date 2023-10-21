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

print(numerical_gradient(function_2, np.array([3.0,4.0])))
# 6.00000000000037801 이런식으로 나오나 간추려서 [6., 8.]으로 출력됨
# 이는 넘파이 배열을 출력할 때 수치를 보기 쉽도록 가공하기 때문
print(numerical_gradient(function_2, np.array([0.0,2.0])))
print(numerical_gradient(function_2, np.array([3.0,0.0])))

# 기울기가 가리키는 쪽은 각 장소에서 함수의 출력 값을 가장 줄이는 방향