from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.core import Dense, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
from time import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import data_provider as dp
import notifier as nt
import tensorflow as tf

config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.75
session = tf.Session(config=config)

K.set_image_dim_ordering('th')
K.set_session(session)

allowed_characters = dp.english_allowed_characters
data_width         = 128
data_height        = len(allowed_characters)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(1, data_width, data_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Activation('sigmoid'))

optimizer = Adam(lr=0.001, clipvalue=0.5)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

train_data = dp.newsgroups_data(subset='train', data_width=data_width, allowed_characters=allowed_characters)
data_size  = len(train_data)

X = np.array([x.data for x in train_data])
y = np.array([x.a_class for x in train_data])
y = to_categorical(y, num_classes=6)

nt.notify(sender_name='GPU Boi', event_name='Data parsed', event_description='X.shape: ' + str(X.shape))

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
res = model.fit(X, y, batch_size=16, epochs=1000, validation_split=0.2, shuffle=True, callbacks=[tensorboard])

nt.notify(sender_name='GPU Boi', event_name='Finished training', event_description='Come and look! acc -> ' + str(res.history['acc'][len(res.history['acc']) - 1]))

model_json = model.to_json()                # serialize model to JSON
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")              # serialize weights to HDF5
