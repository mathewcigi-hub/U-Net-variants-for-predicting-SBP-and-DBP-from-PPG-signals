from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate, Dropout, BatchNormalization, Conv1DTranspose
from keras.layers import Activation
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf



def fire_module(x, fire_id, squeeze=16, expand=64):
    f_name = "fire{0}/{1}"
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv1D(squeeze, 1, activation='relu', padding='same', name=f_name.format(fire_id, "squeeze1x1"))(x)
    x = BatchNormalization(axis=channel_axis)(x)

    left = Conv1D(expand, 1, activation='relu', padding='same', name=f_name.format(fire_id, "expand1x1"))(x)
    right = Conv1D(expand, 3, activation='relu', padding='same', name=f_name.format(fire_id, "expand1x3"))(x)
    x = concatenate([left, right], axis=channel_axis, name=f_name.format(fire_id, "concat"))
    return x

def SqueezeUNet(input_size=(512, 1), num_classes=None, deconv_ksize=3, dropout=0.5, activation='sigmoid'):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    if num_classes is None:
        num_classes = input_size[channel_axis]

    inputs = Input(shape=input_size)

    # Downsampling path
    x01 = Conv1D(64, 3, padding='same', activation='relu', name='conv1')(inputs)
    x02 = MaxPooling1D(pool_size=2, padding="same")(x01)  # Output: (256, 64)

    x03 = fire_module(x02, fire_id=2, squeeze=16, expand=64)
    x04 = fire_module(x03, fire_id=3, squeeze=16, expand=64)
    x05 = MaxPooling1D(pool_size=2, padding="same")(x04)  # Output: (128, 128)

    x06 = fire_module(x05, fire_id=4, squeeze=32, expand=128)
    x07 = fire_module(x06, fire_id=5, squeeze=32, expand=128)
    x08 = MaxPooling1D(pool_size=2, padding="same")(x07)  # Output: (64, 256)

    x09 = fire_module(x08, fire_id=6, squeeze=48, expand=192)
    x10 = fire_module(x09, fire_id=7, squeeze=48, expand=192)
    x11 = fire_module(x10, fire_id=8, squeeze=64, expand=256)
    x12 = fire_module(x11, fire_id=9, squeeze=64, expand=256)

    if dropout != 0.0:
        x12 = Dropout(dropout)(x12)

    # Upsampling path
    up1 = concatenate([
        UpSampling1D(size=1)(Conv1DTranspose(192, deconv_ksize, padding='same')(x12)),
        x10,
    ], axis=channel_axis)
    up1 = fire_module(up1, fire_id=10, squeeze=48, expand=192)
    

    up2 = concatenate([
        UpSampling1D(size=1)(Conv1DTranspose(128, deconv_ksize, padding='same')(up1)),
        x08,
    ], axis=channel_axis)
    up2 = fire_module(up2, fire_id=11, squeeze=32, expand=128)

    
    up3 = concatenate([
        UpSampling1D(size=2)(Conv1DTranspose(64, deconv_ksize, padding='same')(up2)),
        x05,
    ], axis=channel_axis)
    up3 = fire_module(up3, fire_id=12, squeeze=16, expand=64)

    
    up4 = concatenate([UpSampling1D(size=2)(Conv1DTranspose(32, deconv_ksize, padding='same')(up3)), x02,], axis=channel_axis)
    up4 = fire_module(up4, fire_id=13, squeeze=16, expand=32)

   
    x = concatenate([UpSampling1D(size=2)(Conv1DTranspose(16, deconv_ksize, padding='same')(up4)), x01,], axis=channel_axis)
    #x = concatenate([up4, x01], axis=channel_axis)
    x = Conv1D(64, 3, padding='same', activation='relu')(x)
    x = Conv1D(num_classes, 3, padding='same', activation=activation)(x)

    
    return Model(inputs=inputs, outputs=x)

# Example usage
#model = SqueezeUNet(input_size=(256, 1), num_classes=1)
#model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

