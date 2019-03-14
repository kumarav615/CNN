
from __future__ import print_function
# import plaidml.keras
# plaidml.keras.install_backend()
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

# Load npz file containing image arrays
x_npz = np.load('x_images_arrays.npz')
x = x_npz['arr_0'] # use x_npz.files to find list of files. For this it returns ['arr_0']

# Load binary encoded labels for Lung Infiltrations: 0=Not_infiltration 1=Infiltration
y_npz = np.load('y_infiltration_labels.npz')
y = y_npz['arr_0']

# First split the data in two sets, 80% for training, 20% for Val/Test)
X_train, X_valtest, y_train, y_valtest = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

# Second split the 20% into validation and test sets
X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=1, stratify=y_valtest)

print(type(X_train))
print(np.array(X_train).shape)
print(np.array(X_val).shape)
print(np.array(X_test).shape)

X_train = np.array(X_train)
print(X_train.shape)

img_width, img_height = 128, 128
nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
epochs = 20
batch_size = 561

K.image_data_format()

train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, rotation_range=30)
valtest_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow(np.array(X_train), y_train, batch_size=batch_size)
validation_generator = valtest_datagen.flow(np.array(X_val), y_val, batch_size=batch_size)
test_generator = valtest_datagen.flow(np.array(X_test), y_test, batch_size=batch_size)

print(train_generator[0][0][0])
