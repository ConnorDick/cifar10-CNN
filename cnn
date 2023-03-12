import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.layers.normalization import batch_normalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(25)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Train samples:", x_train.shape, y_train.shape)
print("Test samples:", x_test.shape, y_test.shape)

print("X_train original shape", x_train.shape)
print("y_train original shape", y_train.shape)
print("X_test original shape", x_test.shape)
print("y_test original shape", y_test.shape)

NUM_CLASSES = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer",  "dog", "frog", "horse", "ship", "truck"]

# show random images from train
cols = 8
rows = 2
fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_train))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_train[random_index, :])
        ax.set_title(cifar10_classes[y_train[random_index, 0]])
plt.show()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers import LeakyReLU

#need to change our labels to categorical instead of integer
y_train  = np_utils.to_categorical(y_train, NUM_CLASSES)
y_test  = np_utils.to_categorical(y_test, NUM_CLASSES)

print(y_train[0])
print(y_test[0])

# Three steps to Convolution
# 1. Convolution
# 2. Activation
# 3. Pooling
# Repeat Steps 1,2,3 for adding more hidden layers

model = Sequential()

#Hidden layer 1
model.add(Conv2D(32, (3, 3), input_shape=(32,32,3)))
model.add(LeakyReLU(alpha=0.3))
BatchNormalization(axis=-1)
model.add(Conv2D(32, (3, 3)))
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2,2)))
BatchNormalization(axis=-1)

#Hidden layer 2
model.add(Conv2D(32,(3, 3)))
model.add(LeakyReLU(alpha=0.3))
BatchNormalization(axis=-1)
model.add(Conv2D(32, (3, 3)))
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2,2)))
BatchNormalization(axis=-1)

model.add(Flatten())
# Fully connected layer
BatchNormalization()
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.3))
BatchNormalization()
model.add(Dropout(0.1))
model.add(Dense(10))

# model.add(Convolution2D(10,3,3, border_mode='same'))
# model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))

#One hot encoding used, so we will use categorical cross entropy
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

#x_test.shape
#y_test.shape
model.fit(x_train, y_train, batch_size=128, epochs=3, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test)
print()
print('Test accuracy: ', score[1])

predictions = model.predict(x_test)
classes_x=np.argmax(predictions,axis=1)

predictions = list(classes_x)
actuals = list(y_test)

sub = pd.DataFrame({'Actual': actuals, 'Predictions': predictions})
sub.to_csv('./output_cnn.csv', index=False)
