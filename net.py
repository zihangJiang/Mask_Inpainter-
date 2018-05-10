import os
from keras.layers import Conv2DTranspose, Reshape, UpSampling2D, Conv2D, LeakyReLU, Flatten, Activation, BatchNormalization, Input, add
from keras.layers import Multiply, Subtract,Dropout,MaxPool2D
from keras.models import Model
import keras.backend as K
import numpy as np
import tensorflow as tf

def P_Conv(x, M, filters, kernel_size = 3, strides = 1 ,padding = "same"):
    x = Multiply()([x, M])
    x = Conv2D(filters, kernel_size = kernel_size, strides=strides, padding = padding)(x)
    M = MaxPool2D(pool_size = (kernel_size,kernel_size), strides=strides,padding = padding)(M)
    return (x,M)
    
    
class Net(object):
    def __init__(self, dim=64, gen_model=None, dis_model=None):
        if gen_model is None:
            masked_imgs = Input(shape=(None, None, 3))
            masks = Input(shape = (None, None, 1))
            ones = Input(shape = (None, None, 3))
            # Encoder
            # 128*128*3
            x = Conv2D(64, kernel_size=3, padding="same")(masked_imgs)
            M = MaxPool2D(pool_size = 3, strides = 1, padding = "same")(masks)
            x1=x
            
            x, M = P_Conv(x, M, 64, kernel_size=5, strides=2)

            z=x
            #x=Conv2D(128, kernel_size=3, padding="same")(x)
            #x=Conv2D(64, kernel_size=3, padding="same")(x)
            #x=add([x,z])
            # 64*64*64
            x=LeakyReLU(alpha=0.1)(x)
            x=BatchNormalization(momentum=0.8)(x)
            x, M = P_Conv(x, M, 128, kernel_size=5, strides=2, padding="same")
            x, M = P_Conv(x, M, 128, kernel_size=3, padding="same")
            y=x
            # 32*32*128
            x=LeakyReLU(alpha=0.1)(x)
            x=BatchNormalization(momentum=0.8)(x)
            x, M = P_Conv(x, M, 128, kernel_size=3, strides=2, padding="same")
            x=BatchNormalization(momentum=0.8)(x)
            x, M = P_Conv(x, M, 128, kernel_size=3, padding="same")
            # 16*16*128
            x=LeakyReLU(alpha=0.1)(x)
            x=BatchNormalization(momentum=0.8)(x)
            x, M = P_Conv(x, M, 256, kernel_size=3, padding="same")
            
            # 16*16*256
            x=LeakyReLU(alpha=0.1)(x)
            x=Dropout(0.1)(x)
    
            # Decoder
    
            x=Conv2DTranspose(256, kernel_size=3, strides=2, padding="same")(x)
            x=Activation('relu')(x)
            # 32*32*256
            x=Conv2D(128, kernel_size=3, padding="same")(x)
            x=add([x,y])
            
            x=LeakyReLU(alpha=0.1)(x)
            x=BatchNormalization(momentum=0.8)(x)
            
            # 32*32*128
            x=Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(x)
            x=Activation('relu')(x)
            # 64*64*128
            x=Conv2D(64, kernel_size=5, padding="same")(x)
            # 64*64*64
            x=add([x,z])
            x=LeakyReLU(alpha=0.1)(x)
            x=BatchNormalization(momentum=0.8)(x)
            
            x=Conv2DTranspose(64, kernel_size=3, strides=2, padding="same")(x)
            x=LeakyReLU(alpha=0.1)(x)
            x=Conv2D(64, kernel_size=3, padding="same")(x)
            x=add([x,x1])

            
            
            x=Activation('relu')(x)
            x=BatchNormalization(momentum=0.8)(x)
            
            
            x=Conv2D(3, kernel_size=3, padding="same")(x)
            x=Activation('tanh')(x)

            #inv_masks = Subtract()([ones,masks])
            comp = add([Multiply()([masks, x]),masked_imgs])
            gen_model = Model([masked_imgs, masks,ones], [x, comp])
        self.generator = gen_model
        '''
        if dis_model is None:
            dis_model = Sequential()
            dis_model.add(Conv2D(dim, 5, strides=2, padding='same', data_format='channels_first', input_shape=[3, 64, 64]))
            dis_model.add(LeakyReLU(0.2))
            dis_model.add(Conv2D(dim*2, 5, strides=2))
            dis_model.add(LeakyReLU(0.2))
            dis_model.add(Flatten())
            dis_model.add(Dense(256))
            dis_model.add(LeakyReLU(0.2))
            dis_model.add(Dense(1))
        self.discriminator = dis_model
        '''

    def save_models(self, name, save_dir='save'):
        self.generator.save(os.path.join(save_dir, "generator_{}.h5".format(name)))
        #self.discriminator.save(os.path.join(save_dir, "discriminator_{}.h5".format(name)))

if __name__ == '__main__':
    net = Net()

