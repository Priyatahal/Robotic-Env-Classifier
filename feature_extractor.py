# -*- coding: utf-8 -*-

#%%
# Feature Extraction
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D,LayerNormalization,Lambda #,gelu

def Conv_patct_swin_transformer():
    out_channels = 192
    kernel_size =  (3,3)
    stride = (1,1)
    padding = "valid"
    # input_shape = (224,300)
    model = Sequential()
    
    model.add(Conv2D(out_channels,kernel_size, stride,input_shape=(224,224,3),name='ConvPatch'))
    # model.add(Conv2D(kernel_size, stride,input_shape))
    model.add(LayerNormalization())
    
    
    model.add(Conv2D(out_channels, kernel_size, stride, padding,name='BasicLayer'))
    model.add(LayerNormalization ())
    model.add(Conv2D(out_channels,kernel_size, stride, padding))
    model.add(LayerNormalization ())
    
    
    model.add(Conv2D(out_channels, kernel_size, stride, padding,name='patch_merge'))
    model.add(LayerNormalization ())
    # model.add(gelu)
    model.add(Lambda(lambda x: tf.keras.activations.gelu(x)))
    
    
    model.add(Conv2D(out_channels, kernel_size, stride, padding,name='BasicLayer1'))
    model.add(LayerNormalization ())
    model.add(Conv2D(out_channels, kernel_size, stride, padding))
    model.add(LayerNormalization ())
    
    
    model.add(Conv2D(out_channels, kernel_size, stride, padding,name='BasicLayer2'))
    model.add(LayerNormalization ())
    model.add(Conv2D(3, kernel_size, stride, padding))
    model.add(LayerNormalization ())
    # model.summary()
    return model

