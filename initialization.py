from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt

inc_model = InceptionV3(include_top=False,
                      weights='imagenet',
                      input_shape =((3, 150, 150)))

#аугментацию данных

bottleneck_datagen = ImageDataGenerator(rescale=1. / 255)  # собственно, генератор


print('Naczalo obrobotki fotok')
train_generator = bottleneck_datagen.flow_from_directory('data/img_train/',
                                                         target_size=(150, 150),
                                                         batch_size=32,
                                                         class_mode=None,
                                                         shuffle=False)
validation_generator = bottleneck_datagen.flow_from_directory('data/img_val/',
                                                         target_size=(150, 150),
                                                         batch_size=32,
                                                         class_mode=None,
                                                         shuffle=False)

#Прогонка данных и сохранение в numpy array
print('Saving bn_features_train.npy to bottleneck_features/')
bottleneck_features_train = inc_model.predict_generator(train_generator, 2000)
np.save(open('bottleneck_features/bn_features_train.npy', 'wb'), bottleneck_features_train)

print('Perwyi block end')
print('Saving bn_features_train.npy to bottleneck_features/')
bottleneck_features_validation = inc_model.predict_generator(validation_generator, 2000)
np.save(open('bottleneck_features/bn_features_validation.npy', 'wb'), bottleneck_features_validation)