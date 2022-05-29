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
from keras.layers import TimeDistributed,Subtract,Input
from keras import layers
from keras import Model

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

def lstmslice(x):
    b = keras.backend.squeeze(x, 1)
    # b = keras.backend.squeeze(x,2)
    # b = np.squeeze(x)
    return b

def kslice(x):
    b = keras.backend.squeeze(x, 3)
    # b = keras.backend.squeeze(x,2)
    # b = np.squeeze(x)
    return b

def unslice(x):
    b = keras.backend.expand_dims(x,1)
    # b = np.squeeze(x)
    return b

# Define the linear combination
def LC(x):
    y = K.constant([0, 1, 0, -1, 0, 1], shape=[2, 3])
    return K.dot(x, K.transpose(y))

def DnCNN(x):
    inpt = x
    # 1st layer, Conv+relu
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_initializer='glorot_uniform')(x)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(1):
        x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_initializer='glorot_uniform')(x)
        #x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
        # last layer, Conv
    x = Convolution2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_initializer='glorot_uniform')(x)
    x = Subtract()([inpt, x])  # input - noise
    # model = Model(inputs=inpt, outputs=x)

    return x


# ## Build the nets
# ### CNN2
dr = 0.5 # dropout rate (%)

i = Input(shape=[2, 1024])

amc=Permute((2,1))(i)
# keras.layers.recurrent.LSTM(128,return_sequences=True,input_shape=(128,2),kernel_regularizer=regularizers.l2(0.001)),

amc=keras.layers.recurrent.LSTM(units=128, kernel_initializer='glorot_uniform',return_sequences=True, dropout=0.1)(amc)
#amc=keras.layers.recurrent.LSTM(units=128, kernel_initializer='glorot_uniform',return_sequences=True, dropout=0.1)(amc)

amc = Flatten()(amc)
amc = Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1")(amc)
amc = Dropout(dr)(amc)
amc = Dense(len(classes), kernel_initializer='he_normal', name="dense2")(amc)
amc = Activation('softmax')(amc)
o = Reshape([len(classes)])(amc)


lstm2 = Model(inputs=[i], outputs=[o])
lstm2.compile(loss='categorical_crossentropy', optimizer='adam')
lstm2.summary()

#lstm2.load_weights('/gpu02/qiaojiansen/Complex_dncnn/snapshot/save_dim2_Dncnn_image_2*2048_filters128_sigma2550_epoch101_aug10_2020-11-16-14-53-23/model_70.h5', by_name=True)
#lstm2.load_weights('/gpu02/qiaojiansen/Complex_dncnn/snapshot/save_dim2_Dncnn_image_2*2048_filters128_sigma2550_epoch101_aug10_2020-11-16-14-53-23/model_45.h5', by_name=True)

# lstm2 = keras.Sequential(
#     [
#         Reshape([1]+in_shp, input_shape=in_shp),
#         Lambda(DnCNN),
#         Lambda(kslice),
#         Permute((2,1)),
#         #keras.layers.recurrent.LSTM(128,return_sequences=True,input_shape=(128,2),kernel_regularizer=regularizers.l2(0.001)),
#         keras.layers.recurrent.LSTM(units=128,return_sequences=True,dropout = 0.1),
#         keras.layers.recurrent.LSTM(units=128,return_sequences=True,dropout = 0.1),
#         #LSTM(units=128,return_sequences=True,dropout = 0.008),
#         # LSTM(units=128, initializer= keras.initializers.Orthogonal(gain=1.0, seed=2020), dropout = 0.8, return_sequences=True),
#         #keras.layers.recurrent.LSTM(units=128,recurrent_initializer='orthogonal',recurrent_dropout=0.4,return_sequences=True),
#
#         #LSTM(units=130, return_sequences=True),
#         #LSTM(units=130,return_sequences=True),
#         Flatten(),
#         Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"),
#         Dropout(dr),
#         Dense(len(classes), kernel_initializer='he_normal', name="dense2"),
#         Activation('softmax'),
#         Reshape([len(classes)])
#     ]
# )
# lstm2.compile(loss='categorical_crossentropy', optimizer='adam')
# lstm2.summary()

# ### Parameterize the Training Process
# Number of epochs
epochs = 400
# Training batch size
# batch_size = 1024
batch_size = 512

# ## Train the networks

#train CNN2
savepath = './lstm1_1024/'
if not os.path.exists(savepath):
    os.makedirs(savepath)

filepath = os.path.join(savepath, 'rice_lstm20.h5')
csv_logger = CSVLogger(savepath+'/log.csv', append=True, separator=',')
print(filepath)
start = time.time()
history_lstm2 = lstm2.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    # show_accuracy=False,
    verbose=2,
    validation_data=(X_test, Y_test),
    class_weight='auto',
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),csv_logger,
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    ])
lstm2.load_weights(filepath)
end = time.time()
duration = end - start
print('lstm2 Training time = ' + str(round(duration/60,5)) + 'minutes')


# Plot confusion matrix
test_Y_hat = lstm2.predict(X_test, batch_size=batch_size)
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
print("Overall Accuracy - lstm2: ", cor / (cor+ncor))
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
acc_lstm2 = np.zeros(len(test_snrs))
for s in range(20):

    test_X_i = samples_db[s]
    test_Y_i = truth_labels_db[s]
    
    # estimate classes
    test_Y_i_hat = lstm2.predict(test_X_i)
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
    acc_lstm2[s] = 1.0*cor/(cor+ncor)
# Save results to a pickle file for plotting later

print(acc_lstm2)
np.save(savepath + 'rice_lstn20.npy', acc_lstm2)




