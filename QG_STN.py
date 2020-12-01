import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib import cm
import tensorflow as tf
import keras.backend as K
#from data_manager import ClutteredMNIST
#from visualizer import plot_mnist_sample
#from visualizer import print_evaluation
#from visualizer import plot_mnist_grid
import netCDF4
import numpy as np
from keras.layers import Input, Convolution2D, Convolution1D, MaxPooling2D, Dense, Dropout, \
                          Flatten, concatenate, Activation, Reshape, \
                          UpSampling2D,ZeroPadding2D
import keras
from keras.callbacks import History 
history = History()

import keras
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, Concatenate, ZeroPadding2D


__version__ = 0.1  # version of the library


def diff_loss(y_pred, y_true):
  beta = 1
  loss = keras.backend.square(y_pred - y_true) + beta*(y_pred)
  loss = keras.backend.mean(loss,axis=-1)
  return loss

def diff_loss_mask(y_pred, y_true):
  mask_var=keras.backend.zeros((10, 88, 128, 2)) 
  mask_var[:,40:60,:,:].assign(keras.backend.ones(20))

  beta = .1
  loss = (1-beta)*keras.backend.square(y_pred - y_true) + beta * (keras.backend.square(mask_var*y_pred-mask_var*y_true))
  loss = keras.backend.mean(loss,axis=1)
  return loss

def diff_loss_grad(y_pred, y_true):
  beta = 0.01
  dy_true, dx_true = tf.image.image_gradients(y_true)
  dy_pred, dx_pred = tf.image.image_gradients(y_pred)
  loss = keras.backend.square(y_pred - y_true)
  loss = keras.backend.mean(loss,axis=-1)+K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)
  return loss

################################################################################
# Circular Convolutional Layer
def CConv2D(filters, kernel_size, strides=(1, 1), activation='linear', padding='valid', kernel_initializer='glorot_uniform', kernel_regularizer=None):
    def CConv2D_inner(x):
        # padding (see https://www.tensorflow.org/api_guides/python/nn#Convolution)
        in_height = int(x.get_shape()[1])
        in_width = int(x.get_shape()[2])

        if (in_height % strides[0] == 0):
            pad_along_height = max(kernel_size[0] - strides[0], 0)
        else:
            pad_along_height = max(
                kernel_size[0] - (in_height % strides[0]), 0)
        if (in_width % strides[1] == 0):
            pad_along_width = max(kernel_size[1] - strides[1], 0)
        else:
            pad_along_width = max(kernel_size[1] - (in_width % strides[1]), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        # left and right side for padding
        pad_left = Cropping2D(cropping=((0, 0), (in_width-pad_left, 0)))(x)
        pad_right = Cropping2D(cropping=((0, 0), (0, in_width-pad_right)))(x)

        # add padding to incoming image
        conc = Concatenate(axis=2)([pad_left, x, pad_right])

        # top/bottom padding options
        if padding == 'same':
            conc = ZeroPadding2D(padding={'top_pad': pad_top,
                                          'bottom_pad': pad_bottom})(conc)
        elif padding == 'valid':
            pass
        else:
            raise Exception('Padding "{}" does not exist!'.format(padding))

        # perform the circular convolution
        cconv2d = Conv2D(filters=filters, kernel_size=kernel_size,
                         strides=strides, activation=activation,
                         padding='valid',
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer)(conc)

        # return circular convolution layer
        return cconv2d
    return CConv2D_inner

from keras.layers import Input
from keras.models import Model
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dense

from utils import get_initial_weights
from layers import BilinearInterpolation


def stn(input_shape=(88, 128, 1), sampling_size=(11, 16), num_classes=10):
    image = Input(shape=input_shape)
    #locnet = Conv2D(32, (5, 5), padding='same')(image)
    locnet = CConv2D(32, (5, 5), padding='same')(image)

    locnet = Activation('relu')(locnet)
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    #locnet = Conv2D(32, (5, 5), padding='same')(locnet)
    locnet = CConv2D(32, (5, 5), padding='same')(locnet)

    locnet = Activation('relu')(locnet)
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = CConv2D(32, (5, 5), padding='same')(locnet)

    #locnet = Conv2D(20, (5, 5), padding='same')(locnet)
    locnet = Activation('relu')(locnet)
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = Flatten()(locnet)
    locnet = Dense(500)(locnet)
    locnet = Activation('relu')(locnet)
    locnet = Dense(200)(locnet)
    locnet = Activation('relu')(locnet)
    locnet = Dense(100)(locnet)
    locnet = Activation('relu')(locnet)
    locnet = Dense(50)(locnet)
    locnet = Activation('relu')(locnet)
    weights = get_initial_weights(50)
    locnet = Dense(6, weights=weights)(locnet)
    x = BilinearInterpolation(sampling_size)([image, locnet])
    #x = Conv2D(32, (3, 3), padding='same')(x)
    x = CConv2D(32, (5, 5), padding='same')(x)

    x = Activation('relu')(x)
    x = UpSampling2D (size=(2,2))(x)
    #x=  Conv2D(32, (3,3), padding='same')(x)
    x = CConv2D(32, (5, 5), padding='same')(x)

    x = Activation('relu')(x)
    x = UpSampling2D (size=(2,2))(x)
    #x = Conv2D(32, (3,3), padding='same')(x)
    x = CConv2D(32, (5, 5), padding='same')(x)

    x = Activation('relu')(x)
    #x = Conv2D(32, (3,3), padding='same')(x)
    x = CConv2D(32, (5, 5), padding='same')(x)

    x = Activation('relu')(x)
    x = UpSampling2D (size=(2,2))(x)
    #x = Conv2D(2, (3,3), padding='same')(x)
    x = CConv2D(1, (5, 5), padding='same')(x)

    x = Activation('linear')(x)
    return Model(inputs=image, outputs=x)

''' Loading QG anomaly data and normalizing'''
#file=netCDF4.Dataset('/global/cscratch1/sd/ashesh/MyQuota/QG_anomalies_90K_unnormalized.nc')
file=netCDF4.Dataset('QG_120K_fullZ_std.nc')
u=file['data']
print('Mean of Full Z',np.mean(np.asarray(u).flatten()))
std_L1=0.49
std_L2=0.49
print(np.shape(u))
u=np.asarray(u)

''' initialize and recorder dimensions '''

trainN=110000
testN=102
lead=1;
batch_size = 10
num_epochs = 8
pool_size = 2
drop_prob=0.0
conv_activation='relu'
Nlat=88
Nlon=128
n_channels=2

''' Divide training and testing data +  account for spin off '''

x_train=u[500:trainN,:,:,0].reshape([109500,88,128,1])
y_train=u[500+lead:trainN+lead,:,:,0].reshape([109500,88,128,1])

print('x_train', np.shape(x_train))
print('y_train', np.shape(y_train))



x_test=(u[trainN+lead+1:trainN+testN,:,:,0]).reshape([100,88,128,1])
y_test=(u[trainN+2*lead+1:trainN+testN+lead,:,:,0]).reshape([100,88,128,1])

print('x_test', np.shape(x_test))
print('y_test', np.shape(y_test))

model = stn()
model.compile(loss=diff_loss_grad, optimizer='adam')
model.summary()

hist = model.fit(x_train, y_train,
                       batch_size = batch_size,
             verbose=1,
             epochs = 100,
             validation_data=(x_test,y_test),shuffle=True,
             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                        min_delta=0,
                                        patience=5, # just to make sure we use a lot of patience before stopping
                                        verbose=0, mode='auto'),
                       keras.callbacks.ModelCheckpoint('best_weights.h5', monitor='val_loss',
                                                    verbose=1, save_best_only=True,
                                                    save_weights_only=True, mode='auto', period=1),history]
             )

print('finished training')

model.load_weights('best_weights.h5')

train_loss=history.history['loss']
val_loss = history.history['val_loss']

plt.plot(train_loss,label='train_loss')
plt.plot(val_loss,label='val_loss')
plt.ylabel('rmse_loss')
plt.xlabel('epochs')
plt.legend()
plt.savefig('loss_with_FullZ.png')


pred=np.zeros([10000,88,128,1])
for k in range (0, 10000):
    if(k==0):
      pred[k,:,:,:]=model.predict(x_test[k,:,:,:].reshape([1,88,128,1]))
    else:
      pred[k,:,:,:]=model.predict(pred[k-1,:,:,:].reshape([1,88,128,1]))    


import scipy
from scipy.io import savemat
savemat('prediction_STN_fullZ_grad_loss_onelayer110K_10Kdays.mat',dict([('prediction',pred),('truth',y_test),('IC',x_test[0,:,:,:])]))




plt.figure(figsize=(20,10))
for k in range(0,5):
 plt.subplot(2,5,k+1)
 plt.contourf(pred[k,:,:,0].squeeze(),cmap=cm.jet)
 plt.clim(-1.0,1.5)
for k in range(0,5):
 plt.subplot(2,5,k+1+5)
 plt.contourf(y_test[k,:,:,0].squeeze(),cmap=cm.jet)
 plt.clim(-1.0,1.5)

plt.savefig('prediction_figure.png') 
