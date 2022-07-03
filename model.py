from __future__ import division
from keras.models import Model
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *





def single_person(frame_l=16, joint_n=15, joint_d=3):

    joints = Input(name='joints', shape=(frame_l, joint_n, joint_d))
    temporal_difference = Input(name='temporal_difference', shape=(frame_l, joint_n, joint_d))
    
    
    ##########   branch 1    ##############
    
    x = Conv2D(filters = 64, kernel_size=(1,1),padding='same')(joints)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(filters = 32, kernel_size=(3,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    

    x = Permute((1,3,2))(x)
    
    
    x = Conv2D(filters = 32, kernel_size=(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)   
    
    
    ##########    branch 1    ##############
    
    
    
    ##########    branch 2 (Temporal difference)   ##############
    
    
    x_d = Conv2D(filters = 64, kernel_size=(1,1),padding='same')(temporal_difference)
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)
    
    x_d = Conv2D(filters = 32, kernel_size=(3,1),padding='same')(x_d)
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)

    x_d = Permute((1,3,2))(x_d)
    
    x_d = Conv2D(filters = 32, kernel_size=(3,3),padding='same')(x_d)
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)
    
    
    ##########     branch 2        ##############
    
    
    # merging Branch1 and Branch2
    x = concatenate([x,x_d], axis=-1)
    
    
    x = Conv2D(filters = 64, kernel_size=(1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x) 
    x = Dropout(0.1)(x)
       
    x = Conv2D(filters = 128, kernel_size=(1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x) 
    x = Dropout(0.1)(x)
      
        
    model = Model(inputs=[joints, temporal_difference], outputs=x)

    return model






def multi_person(max_bodies, frame_l=16, joint_n=15, joint_d=3):
    
    inp, temporal_differences = [], []
    
    for i in range(max_bodies):
        inp.append(Input(name=f'inp_j_{i}', shape=(frame_l, joint_n, joint_d)))
        temporal_differences.append(Input(name=f'inp_j_diff_{i}', shape=(frame_l, joint_n, joint_d)))
    

    single = single_person()


    # Merging conv6 features of all persons
    x = single([inp[0], temporal_differences[0]])
    for i in range(1, max_bodies):
        x = Maximum()([x, single([inp[i], temporal_differences[i]])])

    
    # Final Dense/Fully connected layer
    x = Flatten()(x)
    x = Dropout(0.1)(x)
     
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    
    x = Dense(8, activation='sigmoid')(x)
     
        
      
    inputs = []
    for i, j in zip(inp, temporal_differences):
        inputs.append(i)
        inputs.append(j)
        
    model = Model(inputs, x)
    
    return model


