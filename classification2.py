# -*- coding: utf-8 -*-
"""
Created on Tue May 29 12:39:15 2018

@author: Rudrajit
"""

from keras.layers import Input, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import numpy as np
import scipy.io
import keras.backend as K

data = scipy.io.loadmat('data9.mat')
X_full = data['x']
print(X_full.shape)
X = X_full

Y_full = data['y']
Y = Y_full

model = Sequential()
model.add(Dense(110, input_dim=50, activation='relu',kernel_regularizer=regularizers.l2(0.001), use_bias=False))
#model.add(Dense(10, activation='relu'))
model.add(Dense(20, activation='softmax',kernel_regularizer=regularizers.l2(0.001), use_bias=False))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='wts9.h5',
      monitor='loss', verbose=1, save_best_only=True)
#monitor='loss'

#model.load_weights('wts6.h5')

model.fit(X, Y, epochs=200, batch_size = 20, validation_split=0, callbacks=[checkpointer])

get_1st_layer_output = K.function([model.layers[0].input],
                                  [model.layers[1].output])
Z = get_1st_layer_output([X])
#print(Z.shape)
scipy.io.savemat('Z9.mat', {'Z':Z})

Y2_pred = model.predict(X)
#print(Y2_pred.shape)
scipy.io.savemat('Y_pred9.mat', {'Y_pred':Y2_pred})

ct = 0
for layer in model.layers:
    weights = layer.get_weights() # list of numpy arrays
    if(ct == 0):
       W1 = weights[0]
       print(W1.shape)
       '''
       W11 = np.transpose(np.reshape(weights[1],[60,1]))
       print(W11.shape)
       W1_aug = np.zeros([31,60])
       W1_aug[0:30,:] = W1
       W1_aug[30,:] = W11
       print(W1_aug.shape)
       '''
       scipy.io.savemat('W1_9.mat', {'W1':W1}) 
       ct = ct+1
    else:
       W2 = weights[0]
       print(W2.shape)
       '''
       W21 = np.transpose(np.reshape(weights[1],[15,1]))
       print(W21.shape)
       W2_aug = np.zeros([61,15])
       W2_aug[0:60,:] = W2
       W2_aug[60,:] = W21
       print(W2_aug.shape)
       '''
       scipy.io.savemat('W2_9.mat', {'W2':W2})

'''
acc = 0 
print(Y.shape)
print(Y2_pred.shape)
for i in range(0,5000):
    idx_1 = np.argmax(Y[i,:])
    idx_2 = np.argmax(Y2_pred[i,:])
    if(idx_1 == idx_2):
        acc=acc+1 
 
print(acc/(5000))
'''