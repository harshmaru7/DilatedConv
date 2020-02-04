

import keras
from keras.layers import Activation, Reshape, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Deconvolution2D
from keras.layers import Conv2DTranspose
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Lambda 
from keras.utils import to_categorical
import tensorflow as tf

from keras.layers import Reshape

from keras import backend as K
from keras import regularizers, optimizers

from keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping,ModelCheckpoint

import scipy.io as scio
import numpy as np    
import os
import matplotlib.pyplot as plt
import math
import re
from scipy.misc import imsave
from scipy import ndimage, misc
from numpy import unravel_index
from operator import sub

def atoi(text) : 
    return int(text) if text.isdigit() else text


def natural_keys(text) :
    return [atoi(c) for c in re.split('(\d+)', text)]


def load_images():
    root_path = ""
    filenames = []
    for root, dirnames, filenames in os.walk("Dataset/Train_images/"):
        filenames.sort(key = natural_keys)
        rootpath = root
    images = []
    for filename in filenames :
        filepath = os.path.join(root,filename)
        image = ndimage.imread(filepath, mode = "L")
        images.append(image)
    return images
        
def load_labels():
    labels = np.load('resized_labels.npy')
    labels_list = []
    for i in range(len(labels)):
        labels_list.append(labels[i])
    return labels_list

images = load_images()
labels_list = load_labels()

print(np.unique(labels_list))

train_labels = np.zeros((770,496,64,8))

for i in range(len(labels_list)) :
    for j in range(496) :
        for k in range(64):
            if(labels_list[i][j][k] == 0):
                train_labels[i][j][k][0] = 1
            if(labels_list[i][j][k] == 1):
                train_labels[i][j][k][1] = 1
            if(labels_list[i][j][k] == 2):
                train_labels[i][j][k][2] = 1
            if(labels_list[i][j][k] == 3):
                train_labels[i][j][k][3] = 1
            if(labels_list[i][j][k] == 4):
                train_labels[i][j][k][4] = 1
            if(labels_list[i][j][k] == 5):
                train_labels[i][j][k][5] = 1
            if(labels_list[i][j][k] == 6):
                train_labels[i][j][k][6] = 1
            if(labels_list[i][j][k] == 7):
                train_labels[i][j][k][7] = 1

images=np.array(images)
images = images.reshape(images.shape[0],496,64,1)

train_indices = np.random.choice(770,500,replace = False)

train_images_random = []
train_labels_random = []

for i in train_indices:
    train_images_random.append(images[i])
    train_labels_random.append(train_labels[i])

test_indices = [x for x in range(770) if x not in train_indices]

test_images = []
test_labels = []
for i in test_indices:
    test_images.append(images[i])
    test_labels.append(train_labels[i])


train_images = np.array(train_images_random)
train_labels = np.array(train_labels_random)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

train_images = train_images.astype('float32')
train_labels = train_labels.astype('float32')
test_images = test_images.astype('float32')
test_labels = test_labels.astype('float32')


data_shape = 496*64

weight_decay = 0.0001

weights = np.load('weighted_cropped_images.npy')
weights_matrix = []
for i in train_indices:
    weights_matrix.append(weights[i])

sample_weights = np.array(weights_matrix)
sample_weights = np.reshape(sample_weights,(500,data_shape))
train_labels = np.reshape(train_labels,(500,data_shape,8))
test_labels = np.reshape(test_labels,(270,data_shape,8))


def get_frontend(input_width,input_height) :
    
    model = Sequential()
    
    model.add(Conv2D(64,(3,3),activation='relu',padding = 'same',name = 'conv1_1',input_shape =(input_width, input_height, 1)))
    model.add(Conv2D(64,(3,3),activation='relu',padding = 'same',name = 'conv1_2'))
    model.add(MaxPooling2D(pool_size=(2,2),name = 'pool_1'))

    
    model.add(Conv2D(128,(3,3),activation='relu',padding = 'same',name = 'conv2_1'))
    model.add(Conv2D(128,(3,3),activation='relu',padding = 'same',name = 'conv2_2'))
    model.add(MaxPooling2D(pool_size=(2,2),name = 'pool_2'))
    
    model.add(Conv2D(256,(3,3),activation='relu',padding = 'same',name = 'conv3_1'))
    model.add(Conv2D(256,(3,3),activation='relu',padding = 'same',name = 'conv3_2'))
    model.add(Conv2D(256,(3,3),activation='relu',padding = 'same',name = 'conv3_3'))
    model.add(MaxPooling2D(pool_size=(2,2),name = 'pool_3'))
    
    model.add(Conv2D(512,(3,3),activation='relu',padding = 'same',name = 'conv4_1'))
    model.add(Conv2D(512,(3,3),activation='relu',padding = 'same',name = 'conv4_2'))
    model.add(Conv2D(512,(3,3),activation='relu',padding = 'same',name = 'conv4_3'))
    
    model.add(Conv2D(512,(3,3),activation='relu',dilation_rate= (2,2), padding = 'same',name = 'conv5_1'))
    model.add(Conv2D(512,(3,3),activation='relu',dilation_rate= (2,2), padding = 'same',name = 'conv5_2'))
    model.add(Conv2D(512,(3,3),activation='relu',dilation_rate= (2,2), padding = 'same',name = 'conv5_3'))
    
    
    model.add(Conv2D(4096,(7,7),dilation_rate= (4,4), padding = "same", activation='relu', name = "fc6"))
    model.add(Dropout(0.5))
    model.add(Conv2D(4096,(1,1),activation='relu',padding = "same",name = "fc7"))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(8,(1,1),activation='linear',name = 'fc-final'))
    
    return model

def add_softmax(model) :
    
    _, curr_width, curr_height, curr_channels = model.layers[-1].output_shape
    
    model.add(Reshape((data_shape,8),input_shape = (496,64,8)))
    model.add(Activation('softmax'))
    return model


def add_context(model,no_of_classes) :
 
    model.add(Conv2D(no_of_classes*2,(3,3),padding = "same",activation = 'relu', name = "ct_conv1_1"))
    model.add(Conv2D(no_of_classes*2,(3,3),padding = "same",activation = 'relu', name = "ct_conv1_2"))

    model.add(Conv2D(no_of_classes*4,(3,3),padding = "same",dilation_rate = (2,2),activation = 'relu', name = "ct_conv2_1"))
    model.add(Conv2D(no_of_classes*8,(3,3),padding = "same",dilation_rate = (4,4),activation = 'relu', name = "ct_conv3_1"))
    model.add(Conv2D(no_of_classes*16,(3,3),padding = 'same', dilation_rate = (8,8),activation = 'relu', name = "ct_conv4_1"))
    model.add(Conv2D(no_of_classes*32,(3,3), padding = 'same',dilation_rate = (16,16),activation = 'relu', name = "ct_conv5_1"))
    
    model.add(Conv2D(no_of_classes*32,(3,3),padding = 'same', activation = 'relu', name = "ct_fc1"))
    
    model.add(Deconvolution2D(no_of_classes, kernel_size = (3,3), strides = (2,2), activation = "relu", name = "ct_deconv_1", padding = "same"))
    model.add(Deconvolution2D(no_of_classes, kernel_size = (3,3), strides = (2,2),activation = "relu", name = "ct_deconv_2", padding = "same"))
    model.add(Deconvolution2D(no_of_classes, kernel_size = (3,3), strides = (2,2),activation = "relu", name = "ct_deconv_3", padding = "same"))
    model.add(Conv2D(no_of_classes,(1,1),activation = 'relu', name = "ct_final"))
    return model

# In[42]:
smooth = 1

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def customized_loss(y_true,y_pred):
    return (1*K.categorical_crossentropy(y_true, y_pred))+(0.5*dice_coef_loss(y_true, y_pred))

optimiser = optimizers.Adam(lr = 0.01)

model = get_frontend(496,64)
model = add_context(model,8)
model = add_softmax(model)

model.compile(optimizer=optimiser,loss=customized_loss,metrics=['accuracy',dice_coef],sample_weight_mode='temporal')

#weights_data = np.load("/home/iplab/Desktop/MajorProjectRelayNet/Multi-Scale-Context-Aggregation-by-Dilated-Convolutions/dilation8_pascal_voc.npy", encoding='latin1').item()
#Defining Callback functions which will be called by model during runtime when specified condition satisfies
lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=6, min_lr=0.5e-6)
csv_logger = CSVLogger('Dilated_model_lr_e2_bs_20_epochs_200.csv')
model_chekpoint = ModelCheckpoint("Dilated_model_lr_e2_bs_20_epochs_200.hdf5",monitor = 'val_loss',verbose = 1,save_best_only=True)

model.fit(train_images,train_labels,batch_size=20,epochs=200,validation_data=(test_images,test_labels),sample_weight=sample_weights,callbacks=[lr_reducer, csv_logger,model_chekpoint])





