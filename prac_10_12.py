import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 각 데이터 형상 출력
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

# normalize는 입력 이미지의 픽셀 값을 0.0~1.0 사이의1 값으로 정규화할지를 정함 False로 설정시 입력 이미지의 픽셀 값은 원래 값 그대로 0~255 유지
# flatten은 입력 이미지를 평탄하게, 즉 1차원 배열로 만들지 정함 False로 설정하면 입력 이미지를 1x28x28의 3차원 배열로 True로 설정시 784개의 원소로 이뤄진 1차원 배열로 저장
# one_hot_label은 레이블을 원 핫 인코딩 형태로 저장할지를 정함
# 원 핫 인코딩 형태란 정답을 뜻하는 원소만 1이고 나머지는 모두 0인 배열
# 즉 False시 7이나 2와 같이 숫자 형태의 레이블을 저장하고 True일 때는 레이블을 원 핫 인코딩하여 저장

# 파이썬에는 피클이라는 편리한 기능이 있음 이는 프로그램 실행 중에 특정 객체를 파일로 저장하는 기능이다.
# 저장해둔 pickle 파일을 로드하면 실행 당시의 객체를 즉시 복원할 수 있다.
# MNIST 데이터셋을 읽는 load_mmist() 함수에서도 pickle을 이용

