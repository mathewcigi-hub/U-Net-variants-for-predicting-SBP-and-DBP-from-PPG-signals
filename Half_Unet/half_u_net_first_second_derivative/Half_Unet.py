import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, SeparableConv2D, MaxPooling2D, UpSampling2D, Add, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, multiply, Add, Activation, Dropout


# Define leaky_relu activation function
def leaky_relu(x):
    return tf.keras.layers.LeakyReLU(alpha=0.1)(x) 


def ghost_module_same(inputs, filter_number):
    conv1 = Conv2D(filter_number, 5, padding='same', kernel_initializer='he_normal')(inputs)
       
    act1 = Activation(leaky_relu)(conv1)
    batch1 = BatchNormalization()(act1)
    conv2 = Conv2D(filter_number, 5, padding='same', kernel_initializer='he_normal')(batch1)
    conv2 = Activation(leaky_relu)(conv2)
    
    return concatenate([act1, conv2], axis=-1) 


# For getting smaller dimension
def ghost_module_different(inputs, filter_number):
    conv1 = Conv2D(filter_number, 5, padding='same', kernel_initializer='he_normal')(inputs)
       
    act1 = Activation(leaky_relu)(conv1)
    batch1 = BatchNormalization()(act1)
    conv2 = SeparableConv2D(filter_number, 5, padding='same', kernel_initializer='he_normal')(batch1)
    conv2 = Activation(leaky_relu)(conv2)
    
    return concatenate([act1, conv2], axis=-1)


def attention_block(skip_connection, gating_signal, inter_channel):
    # Apply a 1x1 convolution to the skip connection feature map
    theta_x = Conv2D(inter_channel, 1, strides=(2,1), padding='same')(skip_connection)
    print("\ntheta_x", theta_x.shape)
    
    # Apply a 1x1 convolution to the gating signal feature map
    phi_g = Conv2D(inter_channel, 1, strides=1, padding='same')(gating_signal)
    print("\nphi_g", phi_g.shape)
    
    # Add the two feature maps
    add_xg = Add()([theta_x, phi_g])
    print("\nadd_xg", add_xg.shape)

    # Apply leaky_relu activation
    act_xg = Activation(leaky_relu)(add_xg)

    # Apply a 1x1 convolution, batch normalization, and sigmoid activation
    psi = Conv2D(1, 1, strides=1, padding='same')(act_xg)
    print("\npsi", psi.shape)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)

    # Upsample psi to match the skip connection dimensions
    upsample_psi = UpSampling2D(size=(2, 1))(psi)
 

    print(upsample_psi.shape)

    # Multiply the original skip connection with the attention map
    attention_output = multiply([skip_connection, upsample_psi])

    return attention_output




def half_model_2D(input_size=(512, 3, 1)):
    
    #Encoder
    inputs = Input(shape=input_size, name="signal")
    
    x1 = ghost_module_same(ghost_module_same(inputs, 32),32)
    print("\nx1", x1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 1))(x1)
    print("\npool1", pool1.shape)
    x2 = ghost_module_same(ghost_module_same(pool1, 32), 32) 
    pool2 = MaxPooling2D(pool_size=(2, 1))(x2)   
    print("\npool2", pool2.shape)
    x3 = ghost_module_same(ghost_module_same(pool2, 32), 32)  
    pool3 = MaxPooling2D(pool_size=(2, 1))(x3)  
    print("\npool3", pool3.shape)
    x4 = ghost_module_different(ghost_module_different(pool3, 32), 32)   
    pool4 = MaxPooling2D(pool_size=(2, 1))(x4)
    print("\nx4", x4.shape)
    x5 = ghost_module_different(ghost_module_different(pool4, 32), 32)
    print("\nx5", x5.shape)



    # Decoder
    attn4 = attention_block(skip_connection=x4, gating_signal=x5, inter_channel=64)       ### First Attention block
    attn3 = attention_block(skip_connection=x3, gating_signal=attn4, inter_channel=64)       
    attn2 = attention_block(skip_connection=x2, gating_signal=attn3, inter_channel=64) 
    attn1 = attention_block(skip_connection=x1, gating_signal=attn2, inter_channel=64) 

    print("\n attn4", attn4.shape)
    print("\n attn3", attn3.shape)
    print("\n attn2", attn2.shape)
    print("\n attn1", attn1.shape)


    
    # We want the final shape to be (None, 512, 48, 64)
    target_shape = (512, 3, 64)

    # Calculating the upsampling factors to reach the target shape
    up5_factor = (target_shape[0] // x5.shape[1], target_shape[1] // x5.shape[2])
    up4_factor = (target_shape[0] // x4.shape[1], target_shape[1] // x4.shape[2])
    up3_factor = (target_shape[0] // x3.shape[1], target_shape[1] // x3.shape[2])
    up2_factor = (target_shape[0] // x2.shape[1], target_shape[1] // x2.shape[2])
    attn1_factor = (target_shape[0] // attn1.shape[1], target_shape[1] // attn1.shape[2])

    # Perform upsampling
    #up5 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (16,1))(x5))
    up5 = UpSampling2D(size=up5_factor)(x5)
    up4 = UpSampling2D(size=up4_factor)(attn4)
    up3 = UpSampling2D(size=up3_factor)(attn3)
    up2 = UpSampling2D(size=up2_factor)(attn2)
    attn1 = UpSampling2D(size=attn1_factor)(attn1)
    

    # Now all upsampled layers should have the shape (None, 512, 48, 64)
    upScaled = Add()([up5, up4, up3, up2, attn1])
    print("\nupScaled", upScaled.shape)
    # instead of x1 changed to up5
    upScaled = Dropout(0.3)(upScaled)
    print("\nOutput From upScaled:",upScaled.shape) 
    #all_conv = ghost_module(ghost_module(upScaled, 32), 32)

    #all_conv = ghost_module_output(ghost_module_output(upScaled))
    final_conv = Conv2D(32, 1)(upScaled) 
    #final_conv = Activation(leaky_relu)(final_conv)
    #print("\nOutput From final_conv:",final_conv.shape) 
    
    half_unet_model = Model(inputs, final_conv)
    return half_unet_model


