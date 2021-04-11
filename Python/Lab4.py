# распознование символов MNIST

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from img_loader import load_correct_images, load_incorrect_images
import matplotlib.pyplot as plt

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# train
batch_size = 128
epochs = 3

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.summary()

# evaluate
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

print("Корректные картинки")
for img_entry in load_correct_images():
    img = img_entry['img']
    pred = model.predict(img)
    max_val = 0
    max_idx = 0
    for idx, val in enumerate(pred[0]):
        if val > max_val:
            max_val = val
            max_idx = idx

    max_val_str = "%.4f" % max_val
    info = f'Предсказана цифра: {max_idx} с вероятностью {max_val_str}, На самом деле: {img_entry["label"]}'
    print(info)
    plt.title(info)
    plt.imshow(img_entry['raw'], cmap=plt.cm.binary)
    plt.show()

print("Не корректные картинки")
for img_entry in load_incorrect_images():
    img = img_entry['img']
    pred = model.predict(img)
    max_val = 0
    max_idx = 0
    for idx, val in enumerate(pred[0]):
        if val > max_val:
            max_val = val
            max_idx = idx

    max_val_str = "%.4f" % max_val
    info = f'Предсказана цифра: {max_idx} с вероятностью {max_val_str}, На самом деле: {img_entry["label"]}'
    print(info)
    plt.title(info)
    plt.imshow(img_entry['raw'], cmap=plt.cm.binary)
    plt.show()
