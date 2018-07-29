from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import SGD
from keras.utils import plot_model
import numpy as np

from keras import backend as K
K.set_image_dim_ordering('th')

import data_provider as dp
import notifier as nt
# import plot_learning as pl

allowed_characters = dp.english_allowed_characters
data_width         = 256
data_height        = len(allowed_characters)
train_data         = dp.newsgroups_data(subset='train', data_width=data_width, allowed_characters=allowed_characters)
data_size          = len(train_data)

X = np.array([x.data for x in train_data])
y = np.array([x.a_class for x in train_data])
y = to_categorical(y, num_classes=6)

nt.notify(sender_name='GPU Boi', event_name='Data parsed', event_description='X.shape: ' + str(X.shape))

model = Sequential()
model.add(Conv2D(input_shape=(1, data_width, data_height), filters=3, kernel_size=5))
model.add(Activation('tanh'))
model.add(Conv2D(filters=3, kernel_size=5))
model.add(Activation('tanh'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('tanh'))
model.add(Dense(256))
model.add(Activation('tanh'))
model.add(Dense(6))
model.add(Activation('tanh'))

sgd = SGD(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(model.summary())
# plot = pl.PlotLearning()
res = model.fit(X, y, batch_size=1024, epochs=1000) #, callbacks=[plot])

print(res.history)
nt.notify(sender_name='GPU Boi', event_name='Finished training', event_description='res: ' + str(res.history))
