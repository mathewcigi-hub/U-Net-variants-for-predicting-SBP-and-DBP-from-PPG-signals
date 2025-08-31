from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate, Dropout, BatchNormalization
from keras.layers import Conv1DTranspose, SeparableConv1D, Flatten, Dense, Activation
from keras.optimizers import Adam


def mobile_unet_1d(pretrained_weights=None, input_size=(300, 1)):
    inputs = Input(input_size)
    
    # Encoder
    conv1  = SeparableConv1D(64, 3, activation='relu', padding='same')(inputs)
    conv1  = BatchNormalization()(conv1)
    conv1  = SeparableConv1D(64, 3, activation='relu', padding='same')(conv1)
    conv1  = BatchNormalization()(conv1)
    pool1  = MaxPooling1D(pool_size=2)(conv1)

    conv2  = SeparableConv1D(128, 3, activation='relu', padding='same')(pool1)
    conv2  = BatchNormalization()(conv2)
    conv2  = SeparableConv1D(128, 3, activation='relu', padding='same')(conv2)
    conv2  = BatchNormalization()(conv2)
    pool2  = MaxPooling1D(pool_size=2)(conv2)
    
    conv3  = SeparableConv1D(256, 3, activation='relu', padding='same')(pool2)
    conv3  = BatchNormalization()(conv3)
    conv3  = SeparableConv1D(256, 3, activation='relu', padding='same')(conv3)
    conv3  = BatchNormalization()(conv3)
    pool3  = MaxPooling1D(pool_size=2)(conv3)
    
    conv4  = SeparableConv1D(512, 3, activation='relu', padding='same')(pool3)
    conv4  = BatchNormalization()(conv4)
    conv4  = SeparableConv1D(512, 3, activation='relu', padding='same')(conv4)
    conv4  = BatchNormalization()(conv4)
    pool4  = MaxPooling1D(pool_size=2)(conv4)    
    
    conv5  = SeparableConv1D(1024, 3, activation='relu', padding='same')(pool4)
    conv5  = BatchNormalization()(conv5)
    conv5  = SeparableConv1D(1024, 3, activation='relu', padding='same')(conv5)
    conv5  = BatchNormalization()(conv5)
    drop1 = Dropout(0.5)(conv5)
    
    # Decoder
    conv6  = Conv1DTranspose(512, 3, strides=2, activation='relu', padding='same')(drop1)
    cat6   = concatenate([conv4, conv6], axis = 2)
    conv6  = SeparableConv1D(512, 3, activation='relu', padding='same')(cat6)
    conv6  = BatchNormalization()(conv6)
    conv6  = SeparableConv1D(512, 3, activation='relu', padding='same')(conv6)
    conv6  = BatchNormalization()(conv6)
    
    conv7  = Conv1DTranspose(256, 3, strides=2, activation='relu', padding='same')(conv6)
    cat7   = concatenate([conv3, conv7], axis = 2)
    conv7  = SeparableConv1D(256, 3, activation='relu', padding='same')(cat7)
    conv7  = BatchNormalization()(conv7)
    conv7  = SeparableConv1D(256, 3, activation='relu', padding='same')(conv7)
    conv7  = BatchNormalization()(conv7)
    
    conv8  = Conv1DTranspose(128, 3, strides=2, activation='relu', padding='same')(conv7)
    cat8   = concatenate([conv2, conv8], axis = 2)
    conv8  = SeparableConv1D(128, 3, activation='relu', padding='same')(cat8)
    conv8  = BatchNormalization()(conv8)
    conv8  = SeparableConv1D(128, 3, activation='relu', padding='same')(conv8)    
    conv8  = BatchNormalization()(conv8)
    
    conv9  = Conv1DTranspose(64, 3, strides=2, activation='relu', padding='same')(conv8)
    cat9   = concatenate([conv1, conv9], axis = 2)
    conv9  = SeparableConv1D(64, 3, activation='relu', padding='same')(cat9)
    conv9  = BatchNormalization()(conv9)
    conv9  = SeparableConv1D(64, 3, activation='relu', padding='same')(conv9)        
    conv9  = BatchNormalization()(conv9)
    conv9  = Conv1D(2, 3, activation='relu', padding='same')(conv9)
    drop2 = Dropout(0.5)(conv9)
    conv10 = Conv1D(1, 1, activation='sigmoid')(drop2)
    

    return Model(inputs=inputs, outputs=conv10)