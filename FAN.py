#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:08:04 2020

@author: sariyanidi
"""

from tensorflow import keras
from functools import partial

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, use_bias=False, activation=None, padding="SAME")

class ConvBlock(keras.layers.Layer):
    def __init__(self, num_input_filters, num_filters, **kwargs):
        super().__init__(**kwargs)
        self.num_input_filters = num_filters
        self.num_filters = num_filters
        self.bn1 = keras.layers.BatchNormalization()
        self.conv1 = DefaultConv2D(filters=num_filters/2)
        self.relu1 = keras.layers.ReLU()
        
        self.bn2 = keras.layers.BatchNormalization()
        self.conv2 = DefaultConv2D(filters=int(num_filters/4))
        self.relu2 = keras.layers.ReLU()
        
        self.bn3 = keras.layers.BatchNormalization()
        self.conv3 = DefaultConv2D(filters=int(num_filters/4))
        self.relu3 = keras.layers.ReLU()
        
        self.concatenator = keras.layers.Concatenate()
        
        self.downsample_layers = []
        
        if num_input_filters != num_filters:
            self.downsample_layers = [keras.layers.BatchNormalization(),
                                      keras.layers.ReLU(), DefaultConv2D(num_filters, kernel_size=1)]
    
    def call(self, x):
        residual = x
        
        out1 = self.bn1(x)
        out1 = self.relu1(out1)
        out1 = self.conv1(out1)
        
        out2 = self.bn2(out1)
        out2 = self.relu2(out2)
        out2 = self.conv2(out2)
        
        out3 = self.bn3(out2)
        out3 = self.relu3(out3)
        out3 = self.conv3(out3)
        
        concat = self.concatenator([out1, out2, out3])
        
        for layer in self.downsample_layers:
            residual = layer(residual)
            
        return residual + concat


class HourGlass(keras.layers.Layer):
    
    def __init__(self, num_features, depth, **kwargs):
        super().__init__(**kwargs)
        self.b1s = []
        self.b2s = []
        self.b2pluses = []
        self.b3s = []
        self.num_features = num_features
        self.avg_pool2d = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
        self.upsampler = keras.layers.UpSampling2D()
        self.depth = depth
        
    def _generate_network(self, level):
        self.b1s.append(ConvBlock(3, self.num_features))
        self.b2s.append(ConvBlock(self.num_features, self.num_features))
        
        if level > 1:
            self.generate_network(level -1)
        else:
            self.b2pluses.append(ConvBlock(self.num_features, self.num_features))
        
        self.b3s.append(ConvBlock(self.num_features, self.num_features))
        
    def _forward(self, level, x):
        level0 = level - 1
        
        up1 = x
        up1 = self.b1s[level0](up1)
        
        low1 = self.avg_pool2d(x)
        low1 = self.b2s[level0](low1)
        
        if level > 1:
            low2 = self._forward(level-1, low1)
        else:
            low2 = low1
            low2 = self.b2pluses[level0](low2)
            
        low3 = low2
        low3 = self.b3s(low3)
        
        #scale_factor = 2.0
        up2 = self.upsampler(low3)
        
        return up1+up2
        
    def forward(self, x):
        return self._forward(self.depth, x)
        
        
class FAN(keras.models.Model):

    def __init__(self, num_modules, **kwargs):
        super().__init__(**kwargs)
        self.num_modules = num_modules
        
        self.relu1 = keras.layers.ReLU()
        self.average_pool2d = keras.layers.AveragePooling2D(strides=(2,2))
        
        self.conv1 = keras.layers.Conv2D(64, 7, strides=(2,2), use_bias=True, padding='SAME')
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)
        
        self.ms = []
        self.top_ms = []
        self.conv_lasts = []
        self.bn_ends = []
        self.ls = []
        self.bls = []
        self.als = []
        
        for hg_module in range(self.num_modules):
            self.ms.append(HourGlass(256, 4))
            self.top_ms.append(ConvBlock(256, 256))
            self.conv_lasts.append(keras.layers.Conv2D(256, kernel_size=1, 
                                                       strides=(1,1), padding='valid', 
                                                       activation='relu'))
            
            self.bn_ends.append(keras.layers.BatchNormalization())
            self.ls.append(keras.layers.Conv2D(68, kernel_size=1, strides=(1,1)))
            
            if hg_module < self.num_modules - 1:
                self.bls.append(keras.layers.Conv2D(256, kernel_size=1))
                self.als.append(keras.layers.Conv2D(256, kernel_size=1))
        
    def call(self, x):

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.average_pool2d(self.conv2(x))
        x = self.conv3(x)
        x = self.conv4(x)
                
        previous = x

        outputs = []
        
        for i in range(self.num_modules):
            
            hg = self.ms[i](previous)
            
            ll = hg
            ll = self.top_ms[i](ll)
            ll = self.bn_ends[i](self.conv_lasts[i](ll))
            
            tmp_out = self.ls[i](ll)
            outputs.append(tmp_out)
            
            if i < self.num_modules - 1:
                ll = self.bls[i](ll)
                tmp_out_ = self.als[i](tmp_out)
                previous = previous + ll + tmp_out_
                
        return outputs[-1]

