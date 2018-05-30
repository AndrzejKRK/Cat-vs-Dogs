from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

import numpy as np
import pandas as pd
import h5py

train_data = np.load(open('bottleneck_features/bn_features_train.npy', 'rb'))
train_labels = np.array([0] * 2 + [1] * 2)

validation_data = np.load(open('bottleneck_features/bn_features_validation.npy', 'rb'))
validation_labels = np.array([0] * 2 + [1] * 2)

#Создание FFN массива

fc_model = Sequential()
fc_model.add(Flatten(input_shape=train_data.shape[1:]))
fc_model.add(Dense(64, activation='relu', name='dense_one'))
fc_model.add(Dropout(0.5, name='dropout_one'))
fc_model.add(Dense(64, activation='relu', name='dense_two'))
fc_model.add(Dropout(0.5, name='dropout_two'))
fc_model.add(Dense(1, activation='sigmoid', name='output'))

fc_model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Тенировка

fc_model.fit(train_data, train_labels,
            nb_epoch=50, batch_size=32,
            validation_data=(validation_data, validation_labels))

fc_model.save_weights('bottleneck_features_and_weights/fc_inception_cats_dogs_250.hdf5') # сохраняем веса

print('- Score -')
loss, accuracy = model.evaluate(validation_data, validation_labels)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

