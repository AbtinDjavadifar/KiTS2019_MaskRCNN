#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 19:27:29 2019

@author: abtin
"""

import os
import numpy as np
import nibabel as nib
import scipy.misc
import json                         



data_path = 'C:\Users\Abtin\Desktop\KiTS\dataselected\'
output_path = 'C:\Users\Abtin\Desktop\KiTS\kidneys_train2019\'
case_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
case_folders.sort()


for i in range(len(case_folders)):
    
    case_path = os.path.join(data_path, case_folders[i])
    images_path = os.path.join(case_path, 'imaging.nii.gz')
    annotations_path = os.path.join(case_path, 'segmentation.nii.gz')
    images = nib.load(images_path)
    annotations = nib.load(annotations_path)
# =============================================================================
#     print('image dimension:' , images.shape)
#     print('annotation dimension:' , annotations.shape)
# =============================================================================
    images_data = images.get_fdata()
    annotations_data = annotations.get_fdata()
    
    for j in range(np.shape(images_data)[0]):
        
        scipy.misc.imsave('{}image_{}.jpg'.format(output_path, str(image_id)), images_data[j,:,:])
        mask_image = annotations_data[j,:,:]


with open('kidney_coco.json', 'w') as outfile:  
    json.dump(coco_annotations, outfile)




