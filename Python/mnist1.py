#сверточная НС
#Мы будем использовать сверточную нейронную сеть, которая состоит из 6 слоев:
#
#Слой свертки, 75 карт признаков, размер ядра свертки: 5х5.
#Слой подвыборки, размер пула 2х2.
#Слой свертки, 100 карт признаков, размер ядра свертки 5х5.
#Слой подвыборки, размер пула 2х2.
#Полносвязный слой, 500 нейронов.
#Полносвязный выходной слой, 10 нейронов, которые соответствуют классам рукописных цифр от 0 до 9.
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

# Устанавливаем seed для повторяемости результатов
numpy.random.seed(42)

# Размер изображения
img_rows, img_cols = 28, 28

# Загружаем данные
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#X_train.shape[0] - 60000
#X_test.shape[0] -10000
# Преобразование размерности изображений
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

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

# 1
# model.add(Conv2D(75, kernel_size=(5, 5),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Conv2D(100, (5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

# 2
# Добавляем уровни сети
model.add(Dense(800, input_dim=784, activation="relu", kernel_initializer="normal"))
model.add(Dense(10, activation="softmax", kernel_initializer="normal"))

# Компилируем модель
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())

# Обучаем сеть
model.fit(X_train, Y_train, batch_size=200, epochs=10, validation_split=0.2, verbose=2)

# Оцениваем качество обучения сети на тестовых данных
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))

#runfile('D:/PY/mnist1.py', wdir='D:/PY')
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d_5 (Conv2D)            (None, 24, 24, 75)        1950      
#_________________________________________________________________
#max_pooling2d_5 (MaxPooling2 (None, 12, 12, 75)        0         
#_________________________________________________________________
#dropout_7 (Dropout)          (None, 12, 12, 75)        0         
#_________________________________________________________________
#conv2d_6 (Conv2D)            (None, 8, 8, 100)         187600    
#_________________________________________________________________
#max_pooling2d_6 (MaxPooling2 (None, 4, 4, 100)         0         
#_________________________________________________________________
#dropout_8 (Dropout)          (None, 4, 4, 100)         0         
#_________________________________________________________________
#flatten_3 (Flatten)          (None, 1600)              0         
#_________________________________________________________________
#dense_5 (Dense)              (None, 500)               800500    
#_________________________________________________________________
#dropout_9 (Dropout)          (None, 500)               0         
#_________________________________________________________________
#dense_6 (Dense)              (None, 10)                5010      
#=================================================================
#Total params: 995,060
#Trainable params: 995,060
#Non-trainable params: 0
#_________________________________________________________________
#None
#Train on 48000 samples, validate on 12000 samples
#Epoch 1/10
# - 95s - loss: 0.2577 - acc: 0.9187 - val_loss: 0.0601 - val_acc: 0.9818
#Epoch 2/10
# - 96s - loss: 0.0664 - acc: 0.9804 - val_loss: 0.0415 - val_acc: 0.9878
#Epoch 3/10
# - 93s - loss: 0.0448 - acc: 0.9860 - val_loss: 0.0364 - val_acc: 0.9882
#Epoch 4/10
# - 95s - loss: 0.0383 - acc: 0.9879 - val_loss: 0.0314 - val_acc: 0.9901
#Epoch 5/10
# - 95s - loss: 0.0313 - acc: 0.9897 - val_loss: 0.0303 - val_acc: 0.9909
#Epoch 6/10
# - 95s - loss: 0.0272 - acc: 0.9911 - val_loss: 0.0288 - val_acc: 0.9915
#Epoch 7/10
# - 95s - loss: 0.0242 - acc: 0.9921 - val_loss: 0.0289 - val_acc: 0.9918
#Epoch 8/10
# - 95s - loss: 0.0189 - acc: 0.9942 - val_loss: 0.0262 - val_acc: 0.9932
#Epoch 9/10
# - 93s - loss: 0.0188 - acc: 0.9938 - val_loss: 0.0310 - val_acc: 0.9921
#Epoch 10/10
# - 94s - loss: 0.0173 - acc: 0.9944 - val_loss: 0.0225 - val_acc: 0.9933
#Точность работы на тестовых данных: 99.37%