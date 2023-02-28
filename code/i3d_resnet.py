# -*- coding: utf-8 -*-

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import os
from time import time
import cv2

"""### Build Data Loader"""

from keras.utils import Sequence
from keras.utils import np_utils

class DataGenerator(Sequence):
    """Data Generator inherited from keras.utils.Sequence
    Args: 
        directory: the path of data set, and each sub-folder will be assigned to one class
        batch_size: the number of data points in each batch
        shuffle: whether to shuffle the data per epoch
    Note:
        If you want to load file with other data format, please fix the method of "load_data" as you want
    """
    def __init__(self, directory, batch_size=1, shuffle=True, data_augmentation=True):
        # Initialize the params
        self.batch_size = batch_size
        self.directory = directory
        self.shuffle = shuffle
        self.data_aug = data_augmentation
        # Load all the save_path of files, and create a dictionary that save the pair of "data:label"
        self.X_path, self.Y_dict = self.search_data() 
        # Print basic statistics information
        self.print_stats()
        return None
        
    def search_data(self):
        X_path = []
        Y_dict = {}
        # list all kinds of sub-folders
        self.dirs = sorted(os.listdir(self.directory))
        one_hots = np_utils.to_categorical(range(len(self.dirs)))
        for i,folder in enumerate(self.dirs):
            folder_path = os.path.join(self.directory,folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path,file)
                # append the each file path, and keep its label  
                X_path.append(file_path)
                Y_dict[file_path] = one_hots[i]
        return X_path, Y_dict
    
    def print_stats(self):
        # calculate basic information
        self.n_files = len(self.X_path)
        self.n_classes = len(self.dirs)
        self.indexes = np.arange(len(self.X_path))
        np.random.shuffle(self.indexes)
        # Output states
        print("Found {} files belonging to {} classes.".format(self.n_files,self.n_classes))
        for i,label in enumerate(self.dirs):
            print('%10s : '%(label),i)
        return None
    
    def __len__(self):
        # calculate the iterations of each epoch
        steps_per_epoch = np.ceil(len(self.X_path) / float(self.batch_size))
        return int(steps_per_epoch)

    def __getitem__(self, index):
        """Get the data of each batch
        """
        # get the indexs of each batch
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # using batch_indexs to get path of current batch
        batch_path = [self.X_path[k] for k in batch_indexs]
        # get batch data
        batch_x, batch_y = self.data_generation(batch_path)
        return batch_x, batch_y

    def on_epoch_end(self):
        # shuffle the data at each end of epoch
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_path):
        # load data into memory, you can change the np.load to any method you want
        batch_x = [self.load_data(x) for x in batch_path]
        batch_y = [self.Y_dict[x] for x in batch_path]
        # transfer the data format and take one-hot coding for labels
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y
      
    def normalize(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data-mean) / std
    
    def random_flip(self, video, prob):
        s = np.random.rand()
        if s < prob:
            video = np.flip(m=video, axis=2)
        return video    
    
    def uniform_sampling(self, video, target_frames=32):
        # get total frames of input video and calculate sampling interval 
        len_frames = int(len(video))
        interval = int(np.ceil(len_frames/target_frames))
        # init empty list for sampled video and 
        sampled_video = []
        for i in range(0,len_frames,interval):
            sampled_video.append(video[i])     
        # calculate numer of padded frames and fix it 
        num_pad = target_frames - len(sampled_video)
        padding = []
        if num_pad>0:
            for i in range(-num_pad,0):
                try: 
                    padding.append(video[i])
                except:
                    padding.append(video[0])
            sampled_video += padding     
        # get sampled video
        return np.array(sampled_video, dtype=np.float32)
    
    def random_clip(self, video, target_frames=32):
        start_point = np.random.randint(len(video)-target_frames)
        return video[start_point:start_point+target_frames]
    
    def dynamic_crop(self, video):
        # extract layer of optical flow from video
        opt_flows = video[...,3]
        # sum of optical flow magnitude of individual frame
        magnitude = np.sum(opt_flows, axis=0)
        # filter slight noise by threshold 
        thresh = np.mean(magnitude)
        magnitude[magnitude<thresh] = 0
        # calculate center of gravity of magnitude map and adding 0.001 to avoid empty value
        x_pdf = np.sum(magnitude, axis=1) + 0.001
        y_pdf = np.sum(magnitude, axis=0) + 0.001
        # normalize PDF of x and y so that the sum of probs = 1
        x_pdf /= np.sum(x_pdf)
        y_pdf /= np.sum(y_pdf)
        # randomly choose some candidates for x and y 
        x_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=x_pdf)
        y_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=y_pdf)
        # get the mean of x and y coordinates for better robustness
        x = int(np.mean(x_points))
        y = int(np.mean(y_points))
        # avoid to beyond boundaries of array
        x = max(56,min(x,167))
        y = max(56,min(y,167))
        # get cropped video 
        return video[:,x-56:x+56,y-56:y+56,:]  
    
    def color_jitter(self,video):
        # range of s-component: 0-1
        # range of v component: 0-255
        s_jitter = np.random.uniform(-0.2,0.2)
        v_jitter = np.random.uniform(-30,30)
        for i in range(len(video)):
            hsv = cv2.cvtColor(video[i], cv2.COLOR_RGB2HSV)
            s = hsv[...,1] + s_jitter
            v = hsv[...,2] + v_jitter
            s[s<0] = 0
            s[s>1] = 1
            v[v<0] = 0
            v[v>255] = 255
            hsv[...,1] = s
            hsv[...,2] = v
            video[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return video
        
    def load_data(self, path):
        # load the processed .npy files which have 5 channels (1-3 for RGB, 4-5 for optical flows)
        data = np.load(path, mmap_mode='r')
        data = np.float32(data)
        # sampling 64 frames uniformly from the entire video
        data = self.uniform_sampling(video=data, target_frames=32)
        # whether to utilize the data augmentation
        if  self.data_aug:
            data[...,:3] = self.color_jitter(data[...,:3])
            data = self.random_flip(data, prob=0.5)
        # normalize rgb images and optical flows, respectively
        data[...,:3] = self.normalize(data[...,:3])
        data[...,3:] = self.normalize(data[...,3:])
        return data

"""#3D Resnet18"""

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Add, Multiply, Activation, Dense, Conv3D, ZeroPadding3D, MaxPooling3D, BatchNormalization, AveragePooling3D, Flatten, Dropout
from tensorflow.keras import regularizers
from keras.utils import plot_model
from keras.layers.core import Lambda
from tensorflow.keras.applications.resnet50 import ResNet50

# define a function to add bottleneck
def add_bottle_neck(x, stride_val, filters_1, filters_2, regress_indentity):
    x_identity = x
    x_layer = Conv3D(filters_1, kernel_size=(1, 3, 3), padding='same', strides=stride_val)(x)
    x_layer = BatchNormalization(axis=-1)(x_layer)
    x_layer = Activation(activation='relu')(x_layer)
    x_layer = Conv3D(filters_1, kernel_size=(1, 3, 3), padding='same', strides=(1, 1, 1))(x_layer)
    x_layer = BatchNormalization(axis=-1)(x_layer)
    x_layer = Activation(activation='relu')(x_layer)
    if (regress_indentity == True):
        shortcut_path = Conv3D(filters_2, kernel_size=(1, 1, 1), padding='valid', strides=stride_val)(x_identity)
        shortcut_path = BatchNormalization(axis=-1)(shortcut_path)
        final_layer = Add()([x_layer, shortcut_path])
    else:
        final_layer = Add()([x_layer, x_identity])
    
    final_layer = Activation(activation='relu')(final_layer)
    return final_layer

# define a function to add bottleneck
def add_bottle_neck_3(x, stride_val, filters_1, filters_2, regress_indentity):
    x_identity = x
    x_layer = Conv3D(filters_1, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same', strides=stride_val)(x)
    x_layer = BatchNormalization(axis=-1)(x_layer)
    x_layer = Activation(activation='relu')(x_layer)
    x_layer = Conv3D(filters_1, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same', strides=(1, 1, 1))(x_layer)
    x_layer = BatchNormalization(axis=-1)(x_layer)
    x_layer = Activation(activation='relu')(x_layer)
    if (regress_indentity == True):
        shortcut_path = Conv3D(filters_2, kernel_size=(1, 1, 1), kernel_initializer='he_normal', padding='valid', strides=stride_val)(x_identity)
        shortcut_path = BatchNormalization(axis=-1)(shortcut_path)
        final_layer = Add()([x_layer, shortcut_path])
    else:
        final_layer = Add()([x_layer, x_identity])
    
    final_layer = Activation(activation='relu')(final_layer)
    return final_layer

def residual_block_4d(x, num_segments):
    main_stream = x
    b,t,h,w,c =x.shape
    x=tf.reshape(x,shape=(-1,num_segments,t,h,w,c))
    x=tf.transpose(x,perm=[0,5,2,3,4,1])
    x=tf.reshape(x,shape=(-1,c,t,h*w,num_segments))
    x=Conv3D(num_segments,kernel_size=(1,3,3), kernel_initializer='he_normal', padding='same')(x)
    x=tf.reshape(x,shape=(-1,c,t,h,w,num_segments))
    x=tf.transpose(x,perm=[0,5,2,3,4,1])
    x_4d=tf.reshape(x,shape=(-1,t,h,w,c))
    x_4d= BatchNormalization(axis=-1)(x_4d)
    x_4d = Activation(activation='relu')(x_4d)
    f_layer=Add()([main_stream, x_4d])
    
    return f_layer

# extract the rgb images 
def get_rgb(input_x):
    rgb = input_x[...,:3]
    return rgb

# extract the optical flows
def get_opt(input_x):
    opt= input_x[...,3:5]
    return opt

inputs = Input(shape=(32,224,224,5))

rgb = Lambda(get_rgb,output_shape=None)(inputs)
opt = Lambda(get_opt,output_shape=None)(inputs)


# define 3d ResNet50
X = ZeroPadding3D((0, 3, 3))(rgb)
X = Conv3D(64, kernel_size=(1, 7, 7,), padding='valid', strides=(1, 2, 2))(X)
X = BatchNormalization(axis=-1)(X)
X = Activation(activation='relu')(X)

X = add_bottle_neck(X, stride_val=(1, 2, 2), filters_1=64, filters_2=64, regress_indentity=True) # conv2_1
X = add_bottle_neck(X, stride_val=(1, 1, 1), filters_1=64, filters_2=64, regress_indentity=False) # conv2_2

X = add_bottle_neck(X, stride_val=(1, 2, 2), filters_1=128, filters_2=128, regress_indentity=True) # conv3_1
X = add_bottle_neck(X, stride_val=(1, 1, 1), filters_1=128, filters_2=128, regress_indentity=False) # conv3_2

X = add_bottle_neck_3(X, stride_val=(1, 2, 2), filters_1=256, filters_2=256, regress_indentity=True) # conv4_1
X = add_bottle_neck_3(X, stride_val=(1, 1, 1), filters_1=256, filters_2=256, regress_indentity=False) # conv4_2

X=residual_block_4d(X,4)

X = add_bottle_neck_3(X, stride_val=(1, 2, 2), filters_1=512, filters_2=512, regress_indentity=True) # conv5_1
X = add_bottle_neck_3(X, stride_val=(1, 1, 1), filters_1=512, filters_2=512, regress_indentity=False) # conv5_2

# X = AveragePooling3D(pool_size=(1,7,7), strides=None, padding='valid')(X) # Modify for glimpse clouds

# ##################################################### Optical Flow channel
opt = Conv3D(
    64, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
opt = Conv3D(
    64, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
opt = MaxPooling3D(pool_size=(1,2,2))(opt)

opt = Conv3D(
    128, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
opt = Conv3D(
    128, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
opt = MaxPooling3D(pool_size=(1,2,2))(opt)

opt = Conv3D(
    256, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
opt = Conv3D(
    256, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
opt = MaxPooling3D(pool_size=(1,2,2))(opt)

opt = Conv3D(
    512, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='sigmoid', padding='same')(opt)
opt = Conv3D(
    512, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='sigmoid', padding='same')(opt)
opt = MaxPooling3D(pool_size=(1,4,4))(opt)


# ##################################################### Fusion and Pooling
# X=tf.reshape(X, shape=tf.shape(opt),name='ConcatResOpt')
x = Multiply()([X,opt])
x = AveragePooling3D(pool_size=(8,1,1), strides=None, padding='valid')(x)
##################################################### Merging Block
x = Conv3D(
    64, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = Conv3D(
    64, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(1,2,2))(x)

x = Conv3D(
    64, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = Conv3D(
    64, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(1,2,2))(x)

x = Conv3D(
    128, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = Conv3D(
    128, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(1,1,1))(x)

##################################################### FC Layers
x = Flatten()(x)
x = Dense(128,activation='relu',kernel_regularizer=regularizers.L2(0.01))(x)
x = Dropout(0.2)(x)
x = Dense(32,activation='relu',kernel_regularizer=regularizers.L2(0.01))(x)

# Build the model
pred = Dense(2, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=pred, name='V4D_ResNet18_3D')

# X = Flatten()(X)
# X = Dense(1000, activation='softmax')(X)

# # Create model
# resnet50_3d_model = Model(inputs = input_x, outputs = X, name='ResNet50_3D')

model.summary()

"""- init data generator"""
num_epochs = 60
batch_size=8

dataset = 'Dataset'

train_generator = DataGenerator(directory='./{}/train'.format(dataset), 
                                batch_size=batch_size, 
                                data_augmentation=True)

val_generator = DataGenerator(directory='./{}/val'.format(dataset),
                              batch_size=batch_size, 
                              data_augmentation=False)


    
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics="accuracy")

from keras.callbacks import ModelCheckpoint, CSVLogger
import keras

class MyCbk(keras.callbacks.Callback):

    def __init__(self, model):
         self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save('Logs_V4DResnet/model_at_epoch_%d.h5' % (epoch+1))

check_point = MyCbk(model)


filename = 'Logs_V4DResnet/ours1_log.csv'
csv_logger = CSVLogger(filename, separator=',', append=True)

callbacks_list = [check_point, csv_logger]

history = model.fit_generator(
    generator=train_generator, 
    validation_data=val_generator,
    callbacks=callbacks_list,
    verbose=1, 
    epochs=num_epochs,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator))