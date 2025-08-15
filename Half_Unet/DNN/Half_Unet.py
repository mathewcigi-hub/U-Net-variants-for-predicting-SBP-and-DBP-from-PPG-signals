

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, SeparableConv1D, MaxPooling1D, UpSampling1D, Add, concatenate
from tensorflow.keras.models import Model

def ghost_module(inputs):
    conv1 = Conv1D(32, 3, padding='same', kernel_initializer='he_normal')(inputs)
    print("\nOutput of first ghost module:",conv1.shape)
    batch1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(batch1)
    conv2 = SeparableConv1D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(act1)
    print("\nOutput of second ghost module:",conv2.shape)
    return concatenate([act1, conv2], axis=-1)  # Concatenate along the last axis for 1D

def half_model(input_size=(256, 1)):
    print("\nInput size:",input_size)
    inputs = Input(shape=input_size, name="signal")

    print("\nInput to X1:",inputs.shape)
    x1 = ghost_module(ghost_module(inputs))
    print("\nOutput From X1:",x1.shape)

    print("\nInput to pool1:",x1.shape)
    pool1 = MaxPooling1D(pool_size=2)(x1)
    print("\nOutput From pool1:",pool1.shape)

    print("\nInput to x2:",pool1.shape)
    x2 = ghost_module(ghost_module(pool1))
    print("\nOutput From x2:",x2.shape)

    print("\nInput to pool2:",x2.shape)
    pool2 = MaxPooling1D(pool_size=2)(x2)
    print("\nOutput From pool2:",pool2.shape)

    print("\nInput to x3:",pool2.shape)
    x3 = ghost_module(ghost_module(pool2))
    print("\nOutput From x3:",x3.shape)

    print("\nInput to pool3:",x3.shape)
    pool3 = MaxPooling1D(pool_size=2)(x3)
    print("\nOutput From pool3:",pool3.shape)

    print("\nInput to x4:",pool3.shape)
    x4 = ghost_module(ghost_module(pool3))
    print("\nOutput From x4:",x4.shape)

    print("\nInput to pool4:",x4.shape)
    pool4 = MaxPooling1D(pool_size=2)(x4)
    print("\nOutput From pool4:",pool4.shape)

    print("\nInput to x5:",pool4.shape)
    x5 = ghost_module(ghost_module(pool4))
    print("\nOutput From x5:",x5.shape)

    # Careful UpSampling for 1D
    up5 = UpSampling1D(size=input_size[0] // x5.shape[1])(x5)
    up4 = UpSampling1D(size=input_size[0] // x4.shape[1])(x4)
    up3 = UpSampling1D(size=input_size[0] // x3.shape[1])(x3)
    up2 = UpSampling1D(size=input_size[0] // x2.shape[1])(x2)

    upScaled = Add()([x1, up2, up3, up4, up5]) 
    print("\nOutput From upScaled:",upScaled.shape) 
    all_conv = ghost_module(ghost_module(upScaled))
    final_conv = Conv1D(1, 1, activation='sigmoid')(all_conv)  # Adjust activation based on your task
    print("\nOutput From final_conv:",final_conv.shape) 
    half_unet_model = Model(inputs, final_conv)
    return half_unet_model


