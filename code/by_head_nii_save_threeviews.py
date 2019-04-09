#!/usr/bin/env python

# Script for assembling images from three anatomical views into one 3D volume.

import nibabel as nib
import numpy as np
import glob
import os
import sys

os.system('rm -rf build')
os.system('mkdir build')

FILE=open('test_id.txt','r')


count_view=4

for line in FILE:
    line=line.rstrip()
    n1_img = nib.load('/data/HGG/'+line+'/'+line+'_flair.nii.gz')
    affine=n1_img.affine

    shape=n1_img.shape

### get a 4 dimension matrix, with the shape size1, size2, size3, channel 

    assembled_matrix=np.zeros((shape[0],shape[1],shape[2],3))

### get the 3rd axis view (axial)
    i=0
    while (i<shape[2]):
        the_current=np.load('result_3rdaxis/'+line+'/'+str(i)+'.npy')
        assembled_matrix[:,:,i,0]+=the_current[:,:,0]*2
        assembled_matrix[:,:,i,1]+=the_current[:,:,1]*2
        assembled_matrix[:,:,i,2]+=the_current[:,:,2]*2
        i=i+1

### get the 1st axis view (sagittal)
    i=0
    while (i<shape[0]):
        the_current=np.load('result_1staxis/'+line+'/'+str(i)+'.npy')
        assembled_matrix[i,:,:,0]+=the_current[:,:,0]
        assembled_matrix[i,:,:,1]+=the_current[:,:,1]
        assembled_matrix[i,:,:,2]+=the_current[:,:,2]
        i=i+1

### get the 2nd axis view (coronal)
    i=0
    while (i<shape[1]):
        the_current=np.load('result_2ndaxis/'+line+'/'+str(i)+'.npy')
        assembled_matrix[:,i,:,0]+=the_current[:,:,0]
        assembled_matrix[:,i,:,1]+=the_current[:,:,1]
        assembled_matrix[:,i,:,2]+=the_current[:,:,2]
        i=i+1




    assembled_matrix=assembled_matrix/count_view
    print(assembled_matrix.max())

### get the assembled max index and save
    index_max=np.argmax(assembled_matrix,axis=3) ## take the max according to view 3
    index_max[index_max==2]=4
    index_max[index_max==1]=2
    index_max[index_max==0]=1
    value_max=np.max(assembled_matrix,axis=3) ## take the maximal value
    #print(value_max.max())
    index_max[value_max<127.5]=0
    img = nib.Nifti1Image(index_max, affine)
    nib.save(img, os.path.join('build',(line+'.nii.gz')))

