# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:45:46 2020

@author: QJS
"""

import os, random
import numpy as np
#from attention import Attention

os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten, Lambda, Permute
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')
import keras
import pickle, random, time
from keras.layers import LSTM, CuDNNLSTM, BatchNormalization,Input
from keras.layers import TimeDistributed, Subtract
from keras import layers
from keras import Model
from keras.layers.noise import AlphaDropout
import tensorflow as tf

# K.set_floatx('float64')

# ## Load the dataset
# - data was downloaded from https://www.deepsig.io/datasets

# Xd = pickle.load(open("./dataset/my.pkl", 'rb'), encoding='latin1')
Xd = pickle.load(open("/gpu01/qiaojiansen/2020_12_23/dataset/rice_alldb_1k/rice_x_h_n_1024_alldb_1000.pkl", 'rb'), encoding='latin1')
test_snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []

for mod in mods:
    for snr in test_snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
X = np.vstack(X)
print(X.shape)

# ### Partition Data
np.random.seed(2019)
n_examples = X.shape[0]
n_train = int(round(n_examples * 0.5))
train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
test_idx = list(set(range(0, n_examples)) - set(train_idx))
X_train = X[train_idx]
X_test = X[test_idx]


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1  # ?
    return yy1


Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods

def kslice(x):
    b = keras.backend.squeeze(x,3)
    # b = np.squeeze(x)
    return b
def unslice(x):
    b = keras.backend.expand_dims(x,1)
    # b = np.squeeze(x)
    return b


def DnCNN(x):
    inpt = x
    # 1st layer, Conv+relu
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer='glorot_uniform', data_format='channels_first')(x)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(9):
        x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          kernel_initializer='glorot_uniform', data_format='channels_first')(x)
        # x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
        # last layer, Conv
    x = Convolution2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer='glorot_uniform')(x)
    x = Subtract()([inpt, x])  # input - noise
    # model = Model(inputs=inpt, outputs=x)

    return x


# Resnet Architecture
# why do they not use batchnorm?
def residual_stack(x):
    def residual_unit(y, _strides=1):
        shortcut_unit = y
        # 1x1 conv linear
        y = layers.Conv1D(32, kernel_size=5, data_format='channels_first', strides=_strides, padding='same',
                          activation='relu')(y)
        y = layers.BatchNormalization()(y)
        y = layers.Conv1D(32, kernel_size=5, data_format='channels_first', strides=_strides, padding='same',
                          activation='linear')(y)
        y = layers.BatchNormalization()(y)
        # add batch normalization
        y = layers.add([shortcut_unit, y])
        return y

    x = layers.Conv1D(32, data_format='channels_first', kernel_size=1, padding='same', activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = residual_unit(x)
    x = residual_unit(x)
    # maxpool for down sampling
    x = layers.MaxPooling1D(data_format='channels_first')(x)
    return x


# ## Build the nets
# ### CNN2
dr = 0.5  # dropout rate (%)

i = Input(shape =[2, 1024])

# for amc
x = residual_stack(i)  # output shape (32,64)
x = residual_stack(x)    # out shape (32,32)
x = residual_stack(x)    # out shape (32,16)    # Comment this when the input dimensions are 1/32 or lower
x = residual_stack(x)    # out shape (32,8)     # Comment this when the input dimensions are 1/16 or lower
x = residual_stack(x)    # out shape (32,4)     # Comment this when the input dimensions are 1/8 or lower
x = Flatten()(x)
x = Dense(128,kernel_initializer="he_normal", activation="selu", name="dense1")(x)
x = AlphaDropout(0.1)(x)
x = Dense(128,kernel_initializer="he_normal", activation="selu", name="dense2")(x)
x = AlphaDropout(0.1)(x)
x = Dense(len(classes),kernel_initializer="he_normal", activation="softmax", name="dense3")(x)
x_out = Reshape([len(classes)])(x)
cnn2 = models.Model(inputs=[i], output=[x_out])
cnn2.compile(loss='categorical_crossentropy', optimizer='adam')
cnn2.summary()


# ### Parameterize the Training Process
# Number of epochs
epochs = 500
# Training batch size
# batch_size = 1024
batch_size = 512

# ## Train the networks

save_dir = './resnet_426/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# train CNN2
start = time.time()
filepath = save_dir+'Resnet-epoch45-patience15.wts.h5'
print(filepath)
history_cnn2 = cnn2.fit(X_train,
                        Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        # show_accuracy=False,
                        verbose=2,
                        validation_data=(X_test, Y_test),
                        class_weight='auto',
                        callbacks=[
                            keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                                            save_best_only=True, mode='auto'),
                            keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')
                        ])
cnn2.load_weights(filepath)
end = time.time()
duration = end - start
print('CNN2 Training time = ' + str(round(duration / 60, 5)) + 'minutes')

# Plot confusion matrix
start = time.time()
test_Y_hat = cnn2.predict(X_test, batch_size=batch_size)
end = time.time()
duration = end - start
print('base-resnet testing time = ' + str(duration) + 's')


conf = np.zeros([len(classes), len(classes)])
confnorm = np.zeros([len(classes), len(classes)])
for i in range(0, X_test.shape[0]):
    j = list(Y_test[i, :]).index(1)
    k = int(np.argmax(test_Y_hat[i, :]))
    conf[j, k] = conf[j, k] + 1
for i in range(0, len(classes)):
    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
cor = np.sum(np.diag(conf))
ncor = np.sum(conf) - cor
print("Overall Accuracy - CNN2: ", cor / (cor + ncor))
acc = 1.0 * cor / (cor + ncor)

# ### Accuracy by SNR (Confusion Matrices @ -20 dB and 20 dB)

# create one hot labels
labels_oh = np.eye(11)
samples_db = np.zeros((20, 11000, 2, 1024))
truth_labels_db = np.zeros((20, 11000, 11))

# Pull out the data by SNR
for i in range(len(test_snrs)):
    for j in range(len(mods)):
        samples_db[i, j * 1000:(j + 1) * 1000, :, :] = Xd[(mods[j], test_snrs[i])]
        truth_labels_db[i, j * 1000:(j + 1) * 1000, :] = labels_oh[j]

# Plot confusion matrix
acc_cnn2 = np.zeros(len(test_snrs))
for s in range(20):

    test_X_i = samples_db[s]
    test_Y_i = truth_labels_db[s]

    # estimate classes
    test_Y_i_hat = cnn2.predict(test_X_i)
    conf = np.zeros([len(mods), len(mods)])
    confnorm = np.zeros([len(mods), len(mods)])
    for i in range(0, test_X_i.shape[0]):
        j = list(test_Y_i[i, :]).index(1)
        k = int(np.argmax(test_Y_i_hat[i, :]))
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(mods)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    # print the confusion matrix @ -20dB and 20dB

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    #     print("Overall Accuracy: ", cor / (cor+ncor))
    acc_cnn2[s] = 1.0 * cor / (cor + ncor)
# Save results to a pickle file for plotting later

print(acc_cnn2)
np.save(save_dir+'Resnet-epoch45-patience15.npy', acc_cnn2)





