import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# x = np.array([1.0, 2.0, 3.0])
# y = np.array([2.0,4.0,6.0])
# print(x+y)
# print(x-y)
# print(x*y)
# print(x/y)
#
# print(x/2.0)
# 배열 x와 y의 원소수가 같아야 함. 만약 다를 시 오류 발생
# 이를 element-wise product

# A = np.array([[1,2], [3,4]])
# print(A)
# print(A.shape)
# print(A.dtype)
#
# B = np.array([[3,0], [0,6]])
# print(A+B)
# print(A*B)
#
# print(A)
# print(A*10)

# 1차원 배열은 벡터
# 2차원 배열은 행렬
# 벡터와 행렬을 일반화한 것을 텐서
# 3차원 이상의 배열을 다차원 배열

# 브로드캐스트
# 넘파이에서 형상이 다른 배열끼리도 계산 가능 -> 브로드캐스트
# A = np.array([[1,2],[3,4]])
# B = np.array([10,20])
# print(A*B)

# 원소 접근
# X = np.array([[51,55], [14,19],[0,4]])
# print(X)
# print(X[0])
# print(X[0][1])
#
# for row in X:
#     print(row)
#
# X = X.flatten() # X를 1차원 배열로 변환(평탄화)
# print(X)
# print(X[np.array([0,2,4])]) # 인덱스가 0,2,4
#
# print(X>15)
# print(X[X>15])

# x = np.arange(0,6,0.1) # 0에서 6까지 0.1 간격으로 생성 [0,0.1,0.2]
# y = np.sin(x)
#
# plt.plot(x,y)
# plt.show()

x = np.arange(0,6,0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x,y1, label="sin")
plt.plot(x,y2, linestyle="--", label="cos")
plt.xlabel("x") # x축 이름
plt.ylabel("y") # y축 이름
plt.title('sin & cos') # 제목
plt.legend() #범례 표시
plt.show()

img = imread('density.jpg')

plt.imshow(img)
plt.show()