#!/usr/bin/env python

# Script for training the U-Net model with 5-fold cross-validation.

from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import keras
import cv2
import sys

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

size= 240
batch_size=16
#ss = 10


mean={}
std={}

# per_head_statistics.txt contains the mean and standard deviation of brain region for each case. 
MEAN_SD_ref=open('/per_head_statistics.txt','r')
for line in MEAN_SD_ref:
    line=line.rstrip()
    table=line.split('\t')
    ttt=table[1].split('.nii.gz')
    mean[ttt[0]]=float(table[2])
    std[ttt[0]]=float(table[3])
MEAN_SD_ref.close()

def preprocess(imgs):
    #imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    #imgs_p = np.ndarray((imgs.shape[0], size), dtype=np.uint8)
    imgs_p=imgs
    #for i in range(imgs.shape[0]):
       # imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)
    #    imgs_p[i] = resize(imgs[i], size, preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

import unet
import random
model = unet.get_unet()
import GS_split

(train_line,test_line)=GS_split.GS_split('train_gs.dat',size)
def generate_data(train_line, batch_size):
    """Replaces Keras' native ImageDataGenerator."""
    i = 0
    while True:
        image_batch = []
        label_batch_0 = []
    #    label_batch_1 = []
    #    label_batch_2 = []
        for b in range(batch_size):
            if i == len(train_line):
                i = 0
                random.shuffle(train_line)
            sample = train_line[i]
            i += 1
            t=sample.split('\t')
            image=np.zeros((size,size,4))
            image1 = np.load(t[0])
            abcd=t[0].split('/')
            nnn=abcd[-2]+'_flair'
            mmm=mean[nnn]
            sss=std[nnn]
            image1=(image1-mmm)/sss


            filename=t[0]
            filename=filename.replace('flair','t1')
            image2 = np.load(filename)
            nnn=abcd[-2]+'_t1'
            mmm=mean[nnn]
            sss=std[nnn]
            image2=(image2-mmm)/sss


            filename=t[0]
            filename=filename.replace('flair','t2')
            image3 = np.load(filename)
            nnn=abcd[-2]+'_t2'
            mmm=mean[nnn]
            sss=std[nnn]
            image3=(image3-mmm)/sss


            filename=t[0]
            filename=filename.replace('flair','t1ce')
            image4 = np.load(filename)

            nnn=abcd[-2]+'_t1ce'
            mmm=mean[nnn]
            sss=std[nnn]
            image4=(image4-mmm)/sss

            (aaa,bbb)=image1.shape
            image[0:aaa,0:bbb,0]=image1
            image[0:aaa,0:bbb,1]=image2
            image[0:aaa,0:bbb,2]=image3
            image[0:aaa,0:bbb,3]=image4

            rrr_flipup=random.random()
            if (rrr_flipup>0.5):
                image=np.flipud(image)

            rrr_fliplr=random.random()
            if (rrr_fliplr>0.5):
                image=np.fliplr(image)

            image_batch.append(image)

            #label_0=np.zeros((size,size,3))
            label_0=np.zeros((size,size,3))
    #        label_1=np.zeros((size,size,1))
    #        label_2=np.zeros((size,size,1))
            if (t[1] == 'neg'):
                pass
            else:
                a=np.load(t[1])
                label_0[0:aaa,0:bbb,0][a==1]=1
                label_0[0:aaa,0:bbb,1][a==2]=1
                label_0[0:aaa,0:bbb,2][a==4]=1

            if (rrr_flipup>0.5):
                label_0=np.flipud(label_0)

            if (rrr_fliplr>0.5):
                label_0=np.fliplr(label_0)

            label_batch_0.append(label_0)
        label_batch_0=np.array(label_batch_0)
        image_batch=np.array(image_batch)
        yield (image_batch, label_batch_0)

#model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=False)
callbacks = [
    keras.callbacks.TensorBoard(log_dir='./',
    histogram_freq=0, write_graph=True, write_images=False),
    keras.callbacks.ModelCheckpoint(os.path.join('./', "weights.h5"),
    verbose=0, save_weights_only=True)#,monitor='val_loss')
    ]

model.fit_generator(
    generate_data(train_line, batch_size),
    #steps_per_epoch=int(len(train_line) // batch_size), nb_epoch=20,validation_data=generate_data(test_line,batch_size),validation_steps=100,callbacks=callbacks)
    steps_per_epoch=int(len(train_line) // batch_size), nb_epoch=10,validation_data=generate_data(test_line,batch_size),validation_steps=100,callbacks=callbacks)

