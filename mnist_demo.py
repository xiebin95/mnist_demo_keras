#参考https://blog.csdn.net/weixin_41055137/article/details/81071226
# Plot ad hoc mnist instances
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

import numpy  # 导入数据库
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.layers import Dropout
# from tensorflow.keras.utils import np_utils

seed = 7  # 设置随机种子
numpy.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()  # 加载数据

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
# 数据集是3维的向量（instance length,width,height).对于多层感知机，模型的输入是二维的向量，因此这
# 里需要将数据集reshape，即将28*28的向量转成784长度的数组。可以用numpy的reshape函数轻松实现这个过
# 程。

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

#将数据reshape，CNN的输入是4维的张量（可看做多维的向量），
# 第一维是样本规模（N），第二维是像素通道（C），第三维和第四维是长度（H）和宽度（W）。(GPU模型NCHW)（CPU模型NCHW）
# 并将数值归一化和类别标签向量化。


# 给定的像素的灰度值在0-255，为了使模型的训练效果更好，通常将数值归一化映射到0-1。
X_train = X_train / 255
X_test = X_test / 255

# 最后，模型的输出是对每个类别的打分预测，对于分类结果从0-9的每个类别都有一个预测分值，表示将模型
# 输入预测为该类的概率大小，概率越大可信度越高。由于原始的数据标签是0-9的整数值，通常将其表示成#0ne-hot向量。如第一个训练数据的标签为5，one-hot表示为[0,0,0,0,0,1,0,0,0,0]。

# y_train = np_utils.to_categorical(y_train)
# # y_test = np_utils.to_categorical(y_test)

y_train =tf.one_hot(y_train,10)
y_test = tf.one_hot(y_test,10)
num_classes = y_test.shape[1]


# 现在需要做得就是搭建神经网络模型了，创建一个函数，建立含有一个隐层的神经网络。
# define baseline model
def baseline_mode_dense():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# define baseline model
# 第一层是卷积层。该层有32个feature map,或者叫滤波器，作为模型的输入层，接受[pixels][width][height]大小的输入数据。feature map的大小是5*5，其输出接一个‘relu’激活函数。
# 下一层是pooling层，使用了MaxPooling，大小为2*2。
# 下一层是Dropout层，该层的作用相当于对参数进行正则化来防止模型过拟合。
# 接下来是全连接层，有128个神经元，激活函数采用‘relu’。
# 最后一层是输出层，有10个神经元，每个神经元对应一个类别，输出值表示样本属于该类别的概率大小。
def baseline_mode_conv2d():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=( 28, 28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# 型的隐含层含有784个节点，接受的输入长度也是784（28*28），最后用softmax函数将预测结果转换为标签
# 的概率值。
# 将训练数据fit到模型，设置了迭代轮数，每轮200个训练样本，将测试集作为验证集，并查看训练的效果。

# build the model
model = baseline_mode_conv2d()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
