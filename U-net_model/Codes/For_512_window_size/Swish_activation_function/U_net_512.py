from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate, Dropout, Flatten, Dense, Activation
from keras.optimizers import Adam
import tensorflow as tf

def switch_activation(x):
    return x * tf.keras.activations.sigmoid(x)  # Core switch activation formula




def unet_1d(pretrained_weights=None, input_size=(300, 1)):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv1D(32, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Activation(switch_activation)(conv1)
    conv1 = Conv1D(32, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = Activation(switch_activation)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    
    conv2 = Conv1D(64, 3, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Activation(switch_activation)(conv2)
    conv2 = Conv1D(64, 3, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = Activation(switch_activation)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)


    conv3 = Conv1D(128, 3, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Activation(switch_activation)(conv3)
    conv3 = Conv1D(128, 3, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = Activation(switch_activation)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)


    conv4 = Conv1D(256, 3, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Activation(switch_activation)(conv4)
    conv4 = Conv1D(256, 3, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = Activation(switch_activation)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling1D(pool_size=2)(drop4)

    conv5 = Conv1D(512, 3, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Activation(switch_activation)(conv5)
    conv5 = Conv1D(512, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = Activation(switch_activation)(conv5)
    drop5 = Dropout(0.5)(conv5)

    
    # Decoder

    up6 = Conv1D(256, 2, padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(drop5))
    up6 = Activation(switch_activation)(up6)
    merge6 = concatenate([drop4, up6], axis=2)
    conv6 = Conv1D(256, 3, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Activation(switch_activation)(conv6)
    conv6 = Conv1D(256, 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Activation(switch_activation)(conv6)


    up7 = Conv1D(128, 2, padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv6))
    up7 = Activation(switch_activation)(up7)
    merge7 = concatenate([conv3, up7], axis=2)
    conv7 = Conv1D(128, 3, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Activation(switch_activation)(conv7)
    conv7 = Conv1D(128, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Activation(switch_activation)(conv7)


    up8 = Conv1D(64, 2, padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv7))
    up8 = Activation(switch_activation)(up8)
    merge8 = concatenate([conv2, up8], axis=2)
    conv8 = Conv1D(64, 3, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Activation(switch_activation)(conv8)
    conv8 = Conv1D(64, 3, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Activation(switch_activation)(conv8)

    up9 = Conv1D(32, 2, padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv8))
    up9 = Activation(switch_activation)(up9)
    merge9 = concatenate([conv1, up9], axis=2)
    conv9 = Conv1D(32, 3, padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Activation(switch_activation)(conv9)
    conv9 = Conv1D(32, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Activation(switch_activation)(conv9)

    drop5 = Dropout(0.5)(conv9)
    conv10 = Conv1D(1, 3, padding='same', kernel_initializer='he_normal')(drop5)
    conv10 = Activation(switch_activation)(conv10)



    # Flatten and connect to a Dense layer for classification
    #flatten_1 = Flatten()(conv10)
    #dense1 = Dense(256, activation='relu')(flatten)
    #dense1 = Dropout(0.5)(dense1)
    #dense2 = Dense(6, activation='softmax')(dense1)  # 6 output classes for BP classification
    
    model = Model(inputs=inputs, outputs=conv10)
    
    #model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    #if pretrained_weights:
     #   model.load_weights(pretrained_weights)
    
    return model
