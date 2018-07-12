from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import SGD
import numpy as np

import data_provider as dp

allowed_characters = dp.english_allowed_characters
data_width         = 256
data_height        = len(allowed_characters)
train_data         = dp.newsgroups_data(subset='train', data_width=data_width, allowed_characters=allowed_characters)
data_size          = len(train_data)

X = np.array([x.data for x in train_data])
y = np.array([x.a_class for x in train_data])

model = Sequential()
model.add(Conv1D(input_shape=(data_width, data_height), filters=3, kernel_size=5))
model.add(Activation('tanh'))
model.add(Conv1D(filters=3, kernel_size=5))
model.add(Activation('tanh'))
model.add(Conv1D(filters=3, kernel_size=5))
model.add(Activation('tanh'))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.inputs)
print(model.summary())

res = model.fit(X, y, batch_size=1, epochs=1000)
print(res)