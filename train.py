#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sariyanidi
@description: This script trains a keras model for face alignment
"""
import random
from FAN import FAN
from glob import glob
from misc import tfdata_generator
from tensorflow import keras
import tensorflow as tf
import os

# The dataset needs to be at this link. To obtain it, 
# (1) download the data_64.tar.gz file from https://drive.google.com/file/d/1T6W7c-3maHbNk5hXo0to_-Kv6dg6tCMp/view?usp=sharing
# (2) copy the data_64.tar.gz file into the ./data folder
# (3) untar the file by running: 
#         cd ./data 
#         tar -xvf data_64.tar.gz
data_dir = './data/ready_64'

# Parse the images and the lables
impaths = glob('%s/*png' % data_dir)
random.seed(1907)
random.shuffle(impaths)

labels = []
for fi, f in enumerate(impaths):
    labelpath = f.replace(".png", ".npy.gz")
    labels.append(labelpath)

batch_size = 20

# Generate training and validation samples
num_tra = 48000
num_tes = 12000
tra_data = tfdata_generator(impaths[:num_tra], labels[:num_tra], True, batch_size=batch_size)
val_data = tfdata_generator(impaths[num_tra:num_tra+num_tes], labels[num_tra:num_tra+num_tes], False, batch_size=batch_size)

# The file where the weights will be stored. We adopt a two-fold optimization strategy, 
# therefore we have two model names
model_key1 = 'FAN_nadam.h5'
model_key2 = 'FAN_nadam_rmsprop.h5'

if not os.path.exists('models'):
    os.mkdir('models')

model_file1 = 'models/%s' % model_key1
model_file2 = 'models/%s' % model_key2

# Generate model
model = FAN(4)
checkpoint_cb = keras.callbacks.ModelCheckpoint(model_file1, save_best_only=True,
                                                save_freq='epoch', 
                                                verbose=1, 
                                                save_weights_only=True)

early_stopping_cb = keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)
model.compile(loss='mse', optimizer='nadam')

# Conduct first training phase
history1 = model.fit(tf.compat.v1.data.make_one_shot_iterator(tra_data), 
                     steps_per_epoch=num_tra// batch_size, epochs=260, 
                     validation_data=tf.compat.v1.data.make_one_shot_iterator(val_data),
                     validation_steps=num_tes // batch_size,
                     batch_size=batch_size,
                     callbacks = [checkpoint_cb, early_stopping_cb])

model.compile(loss='mse', optimizer='rmsprop')
checkpoint_cb = keras.callbacks.ModelCheckpoint(model_file2, save_best_only=True,
                                                save_freq='epoch', verbose=1, save_weights_only=True)

# Conduct second training phase
history2 = model.fit(tf.compat.v1.data.make_one_shot_iterator(tra_data), 
                     steps_per_epoch=num_tra// batch_size, epochs=400, 
                     validation_data=tf.compat.v1.data.make_one_shot_iterator(val_data),
                     validation_steps=num_tes // batch_size,
                     batch_size=batch_size,
                     callbacks = [checkpoint_cb, early_stopping_cb])
