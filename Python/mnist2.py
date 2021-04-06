# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 19:53:58 2018

@author: Елена
"""
#полносвязная НС

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt

# Устанавливаем seed для повторяемости результатов
numpy.random.seed(42)

# Загружаем данные
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Просмотр изображения
digit = X_train[100]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# Преобразование размерности изображений
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
# Нормализация данных
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Преобразуем метки в категории
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


# Создаем последовательную модель
model = Sequential()

# Добавляем уровни сети
model.add(Dense(800, input_dim=784, activation="relu", kernel_initializer="normal"))
model.add(Dense(10, activation="softmax", kernel_initializer="normal"))



# Компилируем модель
#1 - categorical_crossentropy
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

#2 - СКО mean_squared_error
#model.compile(loss="mean_squared_error", optimizer="SGD", metrics=["accuracy"])

print(model.summary())

# Обучаем сеть
model.fit(X_train, Y_train, batch_size=200, epochs=25, validation_split=0.2, verbose=2)

# Оцениваем качество обучения сети на тестовых данных
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))


#1------------------------------------
#runfile('D:/PY/mnist2.py', wdir='D:/PY')
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#dense_7 (Dense)              (None, 800)               628000    
#_________________________________________________________________
#dense_8 (Dense)              (None, 10)                8010      
#=================================================================
#Total params: 636,010
#Trainable params: 636,010
#Non-trainable params: 0
#_________________________________________________________________
#None
#Train on 48000 samples, validate on 12000 samples
#Epoch 1/25
# - 3s - loss: 1.3220 - acc: 0.6916 - val_loss: 0.7795 - val_acc: 0.8502
#Epoch 2/25
# - 2s - loss: 0.6730 - acc: 0.8522 - val_loss: 0.5376 - val_acc: 0.8732
#Epoch 3/25
# - 2s - loss: 0.5239 - acc: 0.8726 - val_loss: 0.4494 - val_acc: 0.8883
#Epoch 4/25
# - 3s - loss: 0.4561 - acc: 0.8836 - val_loss: 0.4033 - val_acc: 0.8976
#Epoch 5/25
# - 3s - loss: 0.4161 - acc: 0.8912 - val_loss: 0.3734 - val_acc: 0.9034
#Epoch 6/25
# - 2s - loss: 0.3889 - acc: 0.8961 - val_loss: 0.3530 - val_acc: 0.9062
#Epoch 7/25
# - 3s - loss: 0.3686 - acc: 0.9008 - val_loss: 0.3370 - val_acc: 0.9103
#Epoch 8/25
# - 3s - loss: 0.3528 - acc: 0.9039 - val_loss: 0.3249 - val_acc: 0.9113
#Epoch 9/25
# - 3s - loss: 0.3399 - acc: 0.9068 - val_loss: 0.3145 - val_acc: 0.9141
#Epoch 10/25
# - 3s - loss: 0.3289 - acc: 0.9095 - val_loss: 0.3061 - val_acc: 0.9165
#Epoch 11/25
# - 3s - loss: 0.3194 - acc: 0.9123 - val_loss: 0.2985 - val_acc: 0.9170
#Epoch 12/25
# - 2s - loss: 0.3110 - acc: 0.9143 - val_loss: 0.2916 - val_acc: 0.9202
#Epoch 13/25
# - 3s - loss: 0.3035 - acc: 0.9163 - val_loss: 0.2855 - val_acc: 0.9214
#Epoch 14/25
# - 3s - loss: 0.2967 - acc: 0.9180 - val_loss: 0.2795 - val_acc: 0.9235
#Epoch 15/25
# - 3s - loss: 0.2904 - acc: 0.9197 - val_loss: 0.2747 - val_acc: 0.9241
#Epoch 16/25
# - 3s - loss: 0.2844 - acc: 0.9214 - val_loss: 0.2702 - val_acc: 0.9264
#Epoch 17/25
# - 3s - loss: 0.2792 - acc: 0.9231 - val_loss: 0.2655 - val_acc: 0.9273
#Epoch 18/25
# - 3s - loss: 0.2741 - acc: 0.9242 - val_loss: 0.2611 - val_acc: 0.9291
#Epoch 19/25
# - 3s - loss: 0.2692 - acc: 0.9258 - val_loss: 0.2572 - val_acc: 0.9300
#Epoch 20/25
# - 3s - loss: 0.2646 - acc: 0.9271 - val_loss: 0.2533 - val_acc: 0.9307
#Epoch 21/25
# - 3s - loss: 0.2603 - acc: 0.9283 - val_loss: 0.2497 - val_acc: 0.9310
#Epoch 22/25
# - 3s - loss: 0.2560 - acc: 0.9296 - val_loss: 0.2464 - val_acc: 0.9319
#Epoch 23/25
# - 3s - loss: 0.2521 - acc: 0.9305 - val_loss: 0.2432 - val_acc: 0.9326
#Epoch 24/25
# - 3s - loss: 0.2482 - acc: 0.9311 - val_loss: 0.2406 - val_acc: 0.9338
#Epoch 25/25
# - 3s - loss: 0.2446 - acc: 0.9324 - val_loss: 0.2366 - val_acc: 0.9342
#Точность работы на тестовых данных: 93.48%

#2----------------------------------------------------------
#runfile('D:/PY/mnist2.py', wdir='D:/PY')
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#dense_9 (Dense)              (None, 800)               628000    
#_________________________________________________________________
#dense_10 (Dense)             (None, 10)                8010      
#=================================================================
#Total params: 636,010
#Trainable params: 636,010
#Non-trainable params: 0
#_________________________________________________________________
#None
#Train on 48000 samples, validate on 12000 samples
#Epoch 1/25
# - 3s - loss: 0.0913 - acc: 0.1282 - val_loss: 0.0898 - val_acc: 0.1895
#Epoch 2/25
# - 2s - loss: 0.0882 - acc: 0.2286 - val_loss: 0.0868 - val_acc: 0.2690
#Epoch 3/25
# - 2s - loss: 0.0854 - acc: 0.2821 - val_loss: 0.0841 - val_acc: 0.3037
#Epoch 4/25
# - 3s - loss: 0.0829 - acc: 0.3173 - val_loss: 0.0816 - val_acc: 0.3417
#Epoch 5/25
# - 3s - loss: 0.0804 - acc: 0.3582 - val_loss: 0.0791 - val_acc: 0.3897
#Epoch 6/25
# - 3s - loss: 0.0779 - acc: 0.4045 - val_loss: 0.0766 - val_acc: 0.4375
#Epoch 7/25
# - 3s - loss: 0.0755 - acc: 0.4490 - val_loss: 0.0741 - val_acc: 0.4770
#Epoch 8/25
# - 2s - loss: 0.0730 - acc: 0.4901 - val_loss: 0.0715 - val_acc: 0.5193
#Epoch 9/25
# - 3s - loss: 0.0705 - acc: 0.5250 - val_loss: 0.0689 - val_acc: 0.5528
#Epoch 10/25
# - 3s - loss: 0.0680 - acc: 0.5594 - val_loss: 0.0663 - val_acc: 0.5862
#Epoch 11/25
# - 3s - loss: 0.0655 - acc: 0.5942 - val_loss: 0.0638 - val_acc: 0.6184
#Epoch 12/25
# - 3s - loss: 0.0630 - acc: 0.6263 - val_loss: 0.0612 - val_acc: 0.6507
#Epoch 13/25
# - 3s - loss: 0.0606 - acc: 0.6552 - val_loss: 0.0586 - val_acc: 0.6826
#Epoch 14/25
# - 3s - loss: 0.0582 - acc: 0.6801 - val_loss: 0.0561 - val_acc: 0.7103
#Epoch 15/25
# - 3s - loss: 0.0558 - acc: 0.7025 - val_loss: 0.0537 - val_acc: 0.7357
#Epoch 16/25
# - 2s - loss: 0.0536 - acc: 0.7236 - val_loss: 0.0514 - val_acc: 0.7569
#Epoch 17/25
# - 2s - loss: 0.0515 - acc: 0.7419 - val_loss: 0.0492 - val_acc: 0.7758
#Epoch 18/25
# - 3s - loss: 0.0494 - acc: 0.7597 - val_loss: 0.0471 - val_acc: 0.7912
#Epoch 19/25
# - 2s - loss: 0.0475 - acc: 0.7739 - val_loss: 0.0451 - val_acc: 0.8031
#Epoch 20/25
# - 2s - loss: 0.0457 - acc: 0.7850 - val_loss: 0.0433 - val_acc: 0.8126
#Epoch 21/25
# - 2s - loss: 0.0441 - acc: 0.7944 - val_loss: 0.0416 - val_acc: 0.8203
#Epoch 22/25
# - 3s - loss: 0.0425 - acc: 0.8015 - val_loss: 0.0401 - val_acc: 0.8266
#Epoch 23/25
# - 3s - loss: 0.0411 - acc: 0.8081 - val_loss: 0.0386 - val_acc: 0.8323
#Epoch 24/25
# - 2s - loss: 0.0398 - acc: 0.8131 - val_loss: 0.0373 - val_acc: 0.8369
#Epoch 25/25
# - 2s - loss: 0.0386 - acc: 0.8187 - val_loss: 0.0361 - val_acc: 0.8400
#Точность работы на тестовых данных: 83.46%