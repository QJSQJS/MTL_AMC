# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:45:46 2020

@author: QJS
"""

# coding: utf-8

# # Complex-Valued Convolutions for Modulation Recognition using Deep Learning
# - Author: Jakob Krzyston
# - Date: 1/27/2020
# 
# This code based on: https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb

# ## Import Packages

# In[ ]:

import os,random
import numpy as np
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,Lambda,Permute
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')
import keras
import pickle, random, time
from keras.layers import LSTM, CuDNNLSTM, BatchNormalization
from keras.layers import TimeDistributed,Subtract
from keras import layers
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
# ## Load the dataset
# - data was downloaded from https://www.deepsig.io/datasets

Xd = pickle.load(open("/gpu01/qiaojiansen/2020_12_23/dataset/rice_alldb_1k/rice_x_h_n_1024_alldb_1000.pkl", 'rb'), encoding = 'latin1')
test_snrs,mods = map(lambda j: sorted( list( set( map( lambda x: x[j], Xd.keys() ) ) ) ), [1,0])
X = []
lbl = []

for mod in mods:
    for snr in test_snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)
print(X.shape)


# ### Partition Data
np.random.seed(2019)
n_examples = X.shape[0]
n_train    = int(round(n_examples * 0.5))
train_idx  = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx   = list(set(range(0,n_examples))-set(train_idx))
X_train    = X[train_idx]
X_test     = X[test_idx]

def to_onehot(yy):
    yy1 = np.zeros([len(yy) ,max(yy)+1])
    yy1[  np.arange(len(yy)),yy] = 1 # ?
    return yy1
Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods


def DnCNN(x):
    inpt = x
    # 1st layer, Conv+relu
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(1):
        x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        #x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
        # last layer, Conv
    x = Convolution2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Subtract()([inpt, x])  # input - noise
    # model = Model(inputs=inpt, outputs=x)

    return x


# ## Build the nets
# ### CNN2
dr = 0.5 # dropout rate (%)
cnn2 = keras.Sequential(
    [
        Reshape([1]+in_shp, input_shape=in_shp),
        #Lambda(DnCNN),
        ZeroPadding2D((0, 2),data_format='channels_first'),
        Convolution2D(256, (1, 3), padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform', data_format='channels_first'),
        #layers.BatchNormalization(),
        Dropout(dr),
        Convolution2D(80, (2, 1), padding='valid', activation="relu", name="conv2", kernel_initializer='glorot_uniform', data_format='channels_first'),
        #layers.BatchNormalization(),
        Dropout(dr),

        Flatten(),
        Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"),
        #layers.BatchNormalization(),
        
        Dropout(dr),
        Dense(len(classes), kernel_initializer='he_normal', name="dense2"),
        #layers.BatchNormalization(),
        Activation('softmax'),
        Reshape([len(classes)])

    ]
)
cnn2.compile(loss='categorical_crossentropy', optimizer='adam')
cnn2.summary()

#cnn2.compile(loss="binary_crossentropy", optimizer="adadelta")
#cnn2.summary()

# ### Parameterize the Training Process
# Number of epochs
epochs = 400
# Training batch size
# batch_size = 1024
batch_size = 512

# ## Train the networks

#train CNN2
start = time.time()

save_dir = './rice_cnn2_1024_99/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

csv_logger = CSVLogger(save_dir+'/log.csv', append=True, separator=',')
filepath = save_dir + 'rice_cnn20.h5'

history_cnn2 = cnn2.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    # show_accuracy=False,
    verbose=2,
    validation_data=(X_test, Y_test),
    class_weight='auto',
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),csv_logger,
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')
    ])
cnn2.load_weights(filepath)
end = time.time()
duration = end - start
print('CNN2 Training time = ' + str(round(duration/60,5)) + 'minutes')


# Plot confusion matrix
start = time.time()
test_Y_hat = cnn2.predict(X_test, batch_size=batch_size)
end = time.time()
duration = end - start
print('base-CNN2 testing time = ' + str(duration) + 's')

conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
cor = np.sum(np.diag(conf))
ncor = np.sum(conf) - cor
print("Overall Accuracy - CNN2: ", cor / (cor+ncor))
acc = 1.0*cor/(cor+ncor)


# ### Accuracy by SNR (Confusion Matrices @ -20 dB and 20 dB)

# create one hot labels
labels_oh       = np.eye(11)
samples_db      = np.zeros((20, 11000, 2, 1024))
truth_labels_db = np.zeros((20, 11000, 11))

# Pull out the data by SNR
for i in range(len(test_snrs)):
    for j in range(len(mods)):
        samples_db[i, j*1000:(j+1)*1000,:,:]    = Xd[(mods[j],test_snrs[i])]
        truth_labels_db[i, j*1000:(j+1)*1000,:] = labels_oh[j]

# Plot confusion matrix
acc_cnn2 = np.zeros(len(test_snrs))
for s in range(20):

    test_X_i = samples_db[s]
    test_Y_i = truth_labels_db[s]
    
    # estimate classes
    test_Y_i_hat = cnn2.predict(test_X_i)
    conf = np.zeros([len(mods),len(mods)])
    confnorm = np.zeros([len(mods),len(mods)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(mods)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    #print the confusion matrix @ -20dB and 20dB
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
#     print("Overall Accuracy: ", cor / (cor+ncor))
    acc_cnn2[s] = 1.0*cor/(cor+ncor)
# Save results to a pickle file for plotting later
print(acc_cnn2)
np.save(save_dir + 'rice_cnn20.npy', acc_cnn2)





