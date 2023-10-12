import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28,28)
print(img.shape)

img_show(img)

# 주의사항은 flatten=True로 설정해 읽어들인 이미지는 1차원 넘파이 배열로 저장되어 있다는 것
# 그래서 이미지로 표시할 때는 원래 형상인 28x28 크기로 다시 변형해야 함 -> reshape()
# 또한 넘파이로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환해야 하며
# 이미지 변환은 Image.fromarray()가 수행된다.