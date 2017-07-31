import sys, os
from common import lib as l
from common import mnist as m

(x_train, t_train), (x_test, t_test) = m.load_mnist(flatten=True, normalize=False)

#print(x_train.shape)
#print(t_train.shape)
#print(x_test.shape)
#print(t_test.shape)

img = x_train[1]
label = t_train[1]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
print(img.shape)  # (28, 28)

l.img_show(img)
