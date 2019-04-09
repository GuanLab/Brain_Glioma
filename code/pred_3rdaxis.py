#!/usr/bin/env python

# Script for predicting the axial images for testing cases.

import os
import sys
import logging
import numpy as np
import cv2
import time
import scipy.io
import glob
import unet
import tensorflow as tf
from keras import backend as K

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
size=240
file_name='test_gs.dat'
os.system('rm -rf result')
os.system('mkdir result')

from keras.models import load_model
from tensorflow import Graph
from tensorflow import Session


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

model0 = unet.get_unet()
model0.load_weights('weights_0.h5')

model1 = unet.get_unet()
model1.load_weights('weights_1.h5')

model2 = unet.get_unet()
model2.load_weights('weights_2.h5')

model3 = unet.get_unet()
model3.load_weights('weights_3.h5')

model4 = unet.get_unet()
model4.load_weights('weights_4.h5')



FILE=open(file_name,'r')
for line in FILE:
    line=line.rstrip()
    image=np.zeros((size,size,4))
    t=line.split('\t')
    line=t[0]
    label_file=t[1]
    image1 = np.load(line)


    abcd=t[0].split('/')
    nnn=abcd[-2]+'_flair'
    mmm=mean[nnn]
    sss=std[nnn]
    image1=(image1-mmm)/sss



    filename=line
    filename=filename.replace('flair','t1')
    image2 = np.load(filename)


    nnn=abcd[-2]+'_t1'
    mmm=mean[nnn]
    sss=std[nnn]
    image2=(image2-mmm)/sss


    filename=line
    filename=filename.replace('flair','t2')
    image3 = np.load(filename)


    nnn=abcd[-2]+'_t2'
    mmm=mean[nnn]
    sss=std[nnn]
    image3=(image3-mmm)/sss


    filename=line
    filename=filename.replace('flair','t1ce')
    image4 = np.load(filename)

    nnn=abcd[-2]+'_t1ce'
    mmm=mean[nnn]
    sss=std[nnn]
    image4=(image4-mmm)/sss









    if (label_file == 'neg'):
        label=np.zeros((240,240))
    else:
        label=cv2.imread(label_file)
    #print(np.mean(image1),np.mean(image2),np.mean(image3),np.mean(image4))
    image[:,:,0]=image1
    image[:,:,1]=image2
    image[:,:,2]=image3
    image[:,:,3]=image4

    image_batch = []
    image_batch.append(image)
    image_batch=np.array(image_batch)
    #print(image_batch.max(),image_batch.min(),np.mean(image_batch))

    pred0=model0.predict(image_batch)
    pred_new0=np.zeros((size,size,3))
    pred_new0=pred0[0,:,:,:]*255

    pred1=model1.predict(image_batch)
    pred_new1=np.zeros((size,size,3))
    pred_new1=pred1[0,:,:,:]*255

    pred2=model2.predict(image_batch)
    pred_new2=np.zeros((size,size,3))
    pred_new2=pred2[0,:,:,:]*255

    pred3=model3.predict(image_batch)
    pred_new3=np.zeros((size,size,3))
    pred_new3=pred3[0,:,:,:]*255

    pred4=model4.predict(image_batch)
    pred_new4=np.zeros((size,size,3))
    pred_new4=pred4[0,:,:,:]*255
    print(pred_new4[:,:,0].max(),pred_new4[:,:,1].max(),pred_new4[:,:,2].max())
    #print(pred_new.mean(),pred_new.max(),pred_new.min())

    t=line.split('/')
    img_id=t[-1]
    pid=t[-2]
    os.system(('mkdir -p result/'+pid))
    ttt=img_id.split('.npy')

    pred_new=(pred_new0+pred_new1+pred_new2+pred_new3+pred_new4)/5.0
    np.save(('result/'+pid+'/'+ttt[0]),pred_new)

    pred_new=np.floor(pred_new).astype('uint8')

    #print(pred.shape)
    cv2.imwrite(('result/'+pid+'/'+ttt[0]+'.png'),pred_new)
    a=cv2.imread(('result/'+pid+'/'+ttt[0]+'.png'))
#    break

