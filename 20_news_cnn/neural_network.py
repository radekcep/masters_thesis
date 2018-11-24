from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.core import Dense, Activation, Flatten
from keras.callbacks import ModelCheckpoint
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

def set_up_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    session = tf.Session(config=config)

    K.set_image_dim_ordering('th')
    K.set_session(session)

def model(data_width, data_height):
    model = Sequential()

    # Convolution Layers

    model.add(Conv1D(256, 7, input_shape=(data_width, data_height)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))

    model.add(Conv1D(256, 7))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))

    model.add(Conv1D(256, 3))
    model.add(Activation('relu'))

    model.add(Conv1D(256, 3))
    model.add(Activation('relu'))

    model.add(Conv1D(256, 3))
    model.add(Activation('relu'))

    model.add(Conv1D(256, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))

    model.add(Flatten())

    # Dense Layers

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(20))
    model.add(Dropout(0.5))

    model.add(Activation(tf.nn.softmax))

    return model


def data(data_width, allowed_characters, subset):
    train_data = dp.newsgroups_data(subset=subset, data_width=data_width, allowed_characters=allowed_characters)

    X = np.array([x.data for x in train_data])
    y = np.array([x.a_class for x in train_data])
    y = to_categorical(y, num_classes=20)

    nt.notify(sender_name='GPU Boi', event_name='Data parsed', event_description='X.shape: ' + str(X.shape))

    return X, y


def save(model):
    print(model.summary())
    model_json = model.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)


set_up_config()

allowed_characters = dp.english_allowed_characters
data_height = len(allowed_characters)
data_width = 1014

model = model(data_width, data_height)
optimizer = Adam(lr=0.001, clipvalue=0.5)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

save(model)
X, y = data(data_width, allowed_characters, subset='train')
test_data = data(data_width, allowed_characters, subset='test')

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
checkpoint = ModelCheckpoint('model/model-{val_acc:.4f}.hdf5', monitor='val_acc', save_best_only=True, mode='max')

res = model.fit(
    X, y,
    batch_size=128,
    epochs=1000,
    shuffle=True,
    validation_data=test_data,
    callbacks=[
        tensorboard,
        checkpoint
    ]
)

nt.notify(
    sender_name='GPU Boi',
    event_name='Finished training',
    event_description='Come and look! acc -> ' + str(res.history['val_acc'][len(res.history['val_acc']) - 1])
)
