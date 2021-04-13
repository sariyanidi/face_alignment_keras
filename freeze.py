#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:08:04 2020
@author: sariyanidi

@description:   This script takes a trained keras model, and converts it
                into a format that's recognizable by OpenCV
"""
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow as tf
import numpy as np
import os

from FAN import FAN

# The weights file is created by running train.py. The code below if the
# weights file exists. If you don't want to run train.py (takes ~10hrs),
# you can simply download the file from the link below
weights_file = './models/FAN_3d_nadam_rmsprop68.h5'
url_weights_file = "http://www.sariyanidi.com/media/%s" % os.path.basename(weights_file)

if not os.path.exists(weights_file):
    print("""Make sure that there is a trained model file at %s"""  % weights_file)
    print("""You can either download the trained model from %s, 
          or train it yourself by running train.py.""" % url_weights_file)
    exit()

# Create the model and load the weights
model = FAN(4)
model.build(input_shape=(None, 256, 256, 3))
model.compile(loss='mse')
model.load_weights(weights_file)                 
model.predict(np.random.rand(1, 256, 256, 3))
model.trainable = False

# The following lines convert the keras model to a tf model 
model.save('model_FAN_final')
loaded = tf.saved_model.load('model_FAN_final')
infer = loaded.signatures['serving_default']
f = tf.function(infer).get_concrete_function(input_1=tf.TensorSpec(shape=[1, 256, 256, 3], dtype=tf.float32))
f2 = convert_variables_to_constants_v2(f)

graph_def = f2.graph.as_graph_def()

# Export frozen graph, the file models/model_FAN_frozen.pb is the file
# that will be used by OpenCV
with tf.io.gfile.GFile('models/model_FAN_frozen.pb', 'wb') as f:
   f.write(graph_def.SerializeToString())

