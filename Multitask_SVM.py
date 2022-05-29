# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:45:46 2020

@author: QJS
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:45:46 2020

@author: QJS
"""

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
import tensorflow
import os, random
import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
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
from keras.layers import Input, Dense, LSTM, CuDNNLSTM, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, \
    multiply, Subtract, Conv2D,UpSampling2D
from keras import layers
from keras import Model
from keras import regularizers

# ## Load the dataset
# - data was downloaded from https://www.deepsig.io/datasets

##################################### with noise ###############################################
Xd = pickle.load(open("/gpu01/qiaojiansen/2020_12_23/dataset/rice_alldb_1k/rice_x_h_n_1024_alldb_1000.pkl", 'rb'),
                 encoding='latin1')

test_snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []

for mod in mods:
    for snr in test_snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
X = np.vstack(X)
print(X.shape)

dr = 0.5
# ### Partition Data
np.random.seed(2019)
n_examples = X.shape[0]
n_train = int(round(n_examples * 0.5))
train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
test_idx = list(set(range(0, n_examples)) - set(train_idx))
X_train = X[train_idx]
X_test = X[test_idx]

###################################### without noise ######################################
Xd_nonoise = pickle.load(open("/gpu01/qiaojiansen/2020_12_23/dataset/rice_alldb_1k/rice_x_h_1024_alldb_1000.pkl", 'rb'),
                 encoding='latin1')

X_nonoise = []
lbl_nonoise = []

for mod in mods:
    for snr in test_snrs:
        X_nonoise.append(Xd_nonoise[(mod, snr)])
        for i in range(Xd_nonoise[(mod, snr)].shape[0]):  lbl_nonoise.append((mod, snr))
X_nonoise = np.vstack(X_nonoise)
print(X_nonoise.shape)

# ### Partition Data

X_nonoise_train = X_nonoise[train_idx]
X_nonoise_test = X_nonoise[test_idx]

########################################   class label    #########################################

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1  # ?
    return yy1


Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods


# ### Complex
# Define the linear combination
def LC(x):
    y = K.constant([0, 1, 0, -1, 0, 1], shape=[2, 3])
    return K.dot(x, K.transpose(y))


def DnCNN(x):
    inpt = x
    # 1st layer, Conv+relu
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu-1-2-3-4-5-71115
    for i in range(15):
        x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        # x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
        # last layer, Conv
    x = Convolution2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Subtract()([inpt, x])  # input - noise
    # model = Model(inputs=inpt, outputs=x)

    return x


def se_layer(x):
    """
    SE-NET
    :param inputs_tensor:input_tensor.shape=[batchsize,h,w,channels]
    :param ratio:
    :param num:
    :return:
    """
    channels = 128
    ratio = 16
    y = GlobalAveragePooling2D()(x)
    y = Reshape((1, 1, channels))(y)
    y = Convolution2D(channels // ratio, (1, 1), strides=1, name="se_conv1_" + str(1), padding="valid")(y)
    y = Activation('relu', name='se_conv1_relu_' + str(1))(y)
    # Dropout(0.5),
    # BatchNormalization(),
    y = Convolution2D(channels, (1, 1), strides=1, name="se_conv2_" + str(1), padding="valid")(y)
    y = Activation('sigmoid', name='se_conv2_relu_' + str(1))(y)
    # Dropout(0.5),
    # BatchNormalization(),
    output = multiply([x, y])
    return output


def kslice(x):
    b = keras.backend.squeeze(x, 3)
    # b = np.squeeze(x)
    return b


def unslice(x):
    b = keras.backend.expand_dims(x, 1)
    # b = np.squeeze(x)
    return b


# def step_decay(epoch):
#     initial_lr = lr
#     if epoch < 50:
#         lr = initial_lr
#     else:
#         lr = initial_lr / 10
#
#     return lr


i_noise = Input(shape=[2, 1024],name="i_noise")

i_reshape = Reshape(in_shp + [1], input_shape=in_shp, name="dncnn_reshape")(i_noise)

# inpt = Input(shape=(None, None, 1))
# 1st layer, Conv+relu
conv1_1 = Conv2D(64, (1, 3), activation='relu', padding='same', name="down_conv1")(i_reshape)
pool1 = MaxPooling2D((1, 2), padding='same', name="down_pool1")(conv1_1)
conv1_2 = Conv2D(64, (1, 3), activation='relu', padding='same', name="down_conv2")(pool1)
h = MaxPooling2D((1, 2), padding='same', name="down_pool2")(conv1_2)

conv2_1 = Conv2D(64, (1, 3), activation='relu', padding='same', name="up_conv1")(h)
up1 = UpSampling2D((1, 2), name="up_pool1")(conv2_1)
conv2_2 = Conv2D(64, (1, 3), activation='relu', padding='same', name="up_conv2")(up1)
up2 = UpSampling2D((1, 2), name="up_pool2")(conv2_2)
x = Conv2D(1, (1, 3), padding='same', name="up_conv3")(up2)

# last layer, Conv
x = Convolution2D(filters=1, kernel_size=(1, 3), strides=(1, 1), padding='same', name="dncnn_out_conv3")(x)
x = Subtract()([i_reshape, x])  # input - noise

# matrix dim
output_denoise = Lambda(kslice,name="dncnn_out")(x)
m = Lambda(unslice)(output_denoise)

# for amc

################################################## CNN ###################################
amc=ZeroPadding2D((0, 2), data_format='channels_first')(m)
# amc=Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1")(denoise_o)
amc =Convolution2D(256, (1, 3), padding='valid', activation="relu", name="conv1",
                      kernel_initializer='glorot_uniform', data_format='channels_first')(amc)
amc =Dropout(dr)(amc)
amc =Convolution2D(80, (2, 1), padding='valid', activation="relu", name="conv2", kernel_initializer='glorot_uniform',
                      data_format='channels_first')(amc)
amc=Dropout(dr)(amc)
amc=Flatten()(amc)
amc=Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1")(amc)
amc=Dropout(dr)(amc)
amc=Dense(len(classes), kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01), name="dense2")(amc)
amc=Activation('softmax')(amc)
#amc=Activation('linear')(amc)
output_class =Reshape([len(classes)],name='class_out')(amc)
###########################################################################################

# ############################################## Complexcnn ##############################
# m = ZeroPadding2D((1, 2), data_format='channels_first')(m)
# # amc=Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1")(denoise_o)
# amc = Convolution2D(256, (2, 3), padding='valid', activation="relu", name="conv1",
#                     kernel_initializer='glorot_uniform', data_format='channels_first')(m)
# amc = Permute((1, 3, 2))(amc)
# amc = Lambda(LC)(amc)
# amc = Permute((1, 3, 2))(amc)
# amc = Activation('relu')(amc)
# amc = Dropout(dr)(amc)
#
# amc = Convolution2D(80, (2, 3), padding='valid', activation="relu", name="conv2", kernel_initializer='glorot_uniform',
#                     data_format='channels_first')(amc)
# amc = Dropout(dr)(amc)
# amc = Flatten()(amc)
# amc = Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1")(amc)
# amc = Dropout(dr)(amc)
# amc = Dense(len(classes), kernel_initializer='he_normal', name="dense2")(amc)
# amc = Activation('softmax')(amc)
# output_class = Reshape([len(classes)],name='class_out')(amc)
# ###########################################################################################

complex_CNN = Model(inputs=i_noise, outputs=[output_denoise,output_class])

complex_CNN.compile(optimizer='adam',
              loss={'dncnn_out': 'mse',
                    'class_out': 'squared_hinge'},
              loss_weights={'dncnn_out': 1,
                            'class_out': 1})
complex_CNN.summary()

save_dir = './dncnn_svm_1024_softmax_real_1223/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

ckpt = ModelCheckpoint(save_dir+'/model_{epoch:02d}.h5', monitor='val_loss',
                    verbose=0, period=1)#period=args.save_every

csv_logger = CSVLogger(save_dir+'/log.csv', append=True, separator=',')
# lr = LearningRateScheduler(step_decay)

epochs = 400

batch_size = 512

filepath = save_dir + 'dncnn_svm.wts.h5'
print(filepath)

start = time.time()

complex_CNN.fit({'i_noise': X_train},
          {'dncnn_out': X_nonoise_train,
           'class_out': Y_train},
            callbacks = [keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'), csv_logger, keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0,mode='auto')],
          epochs=epochs, batch_size=batch_size, validation_split=0.1)

end = time.time()

#complex_CNN.save_weights(filepath)

complex_CNN.load_weights(filepath)

duration = end - start
print('Complex Training time = ' + str(round(duration / 60, 5)) + 'minutes')
# complex_CNN.load_weights(
#     '/gpu02/qiaojiansen/Complex_dncnn/snapshot/save_dim2_Dncnn_2layer_image_2*2048_filters128_sigma25_epoch71_aug1_2020-12-06-19-05-05/model_70.h5',
#     by_name=True)

# complex_CNN.load_weights('/gpu02/qiaojiansen/Complex_dncnn/snapshot/save_dim2_Dncnn_image_small_2*2048_filters128_sigma2550_epoch101_aug1_2020-11-16-16-40-09/model_60.h5', by_name=True)
# complex_CNN.load_weights('/gpu02/qiaojiansen/Complex_dncnn/snapshot/save_dim2_Dncnn_image_2*2048_filters128_sigma2550_epoch101_aug10_2020-11-16-14-53-23/model_45.h5', by_name=True)


# ### Parameterize the Training Process
# Number of epochs
# epochs = 400
# Training batch size

# batch_size = 2048

# ## Train the networks
# train Complex

# history_complex = complex_CNN.fit({'i_noise': X_train},
#                                   {'output_denoise': X_nonoise_train, 'output_class': Y_train},
#                                   batch_size=batch_size,
#                                   epochs=epochs,
#                                   # show_accuracy=False,
#                                   verbose=2,
#                                   validation_data=(X_test, X_nonoise_test, Y_test),
#                                   class_weight='auto',
#                                   callbacks=[
#                                       keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
#                                                                       save_best_only=True, mode='auto'),
#                                       keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=0,
#                                                                     mode='auto')
#                                   ])


# Plot confusion matrix
start = time.time()
output_denoise,test_Y_hat = complex_CNN.predict(X_test, batch_size=batch_size)
end = time.time()
duration = end - start
print('mutil-svm testing time = ' + str(duration) + 's')

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
print("Overall Accuracy - Complex: ", cor / (cor + ncor))
acc = 1.0 * cor / (cor + ncor)

# ### Accuracy by SNR (Confusion Matrices @ -20 dB and 20 dB)

# create one hot labels
labels_oh       = np.eye(11)
samples_db      = np.zeros((20, 11000, 2, 1024))
truth_labels_db = np.zeros((20, 11000, 11))

# Pull out the data by SNR
for i in range(len(test_snrs)):
    for j in range(len(mods)):
        samples_db[i, j * 1000:(j + 1) * 1000, :, :] = Xd[(mods[j], test_snrs[i])]
        truth_labels_db[i, j * 1000:(j + 1) * 1000, :] = labels_oh[j]

# Plot confusion matrix
acc_complex = np.zeros(len(test_snrs))
for s in range(20):

    # extract classes @ SNR
    #     test_SNRs = map(lambda x: lbl[x][1], test_idx)
    test_X_i = samples_db[s]
    test_Y_i = truth_labels_db[s]

    # estimate classes
    output_i_denoise,test_Y_i_hat = complex_CNN.predict(test_X_i)
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
    acc_complex[s] = 1.0 * cor / (cor + ncor)
# Save results to a pickle file for plotting later

print(acc_complex)
np.save(save_dir + 'acc_dncnn_svm.npy',acc_complex)


