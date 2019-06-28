
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
import numpy as np
# =============================================================================
# import json                         
# =============================================================================


data_path = '/home/abtin/kits19/train/dataselected/'
images_path = '/home/abtin/kits19/train/kidneys_train2019/'
annotations_path = '/home/abtin/kits19/train/annotations/'
case_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
case_folders.sort()


for i in range(len(case_folders)):
    
    case_path = os.path.join(data_path, case_folders[i])
    case_images_path = os.path.join(case_path, 'imaging.nii.gz')
    case_annotations_path = os.path.join(case_path, 'segmentation.nii.gz')
    images = nib.load(case_images_path)
    annotations = nib.load(case_annotations_path)
    images_data = images.get_fdata()
    annotations_data = annotations.get_fdata()
    
    for j in range(np.shape(images_data)[0]):

        background = (annotations_data[j,:,:] == 0).astype(int)*255
        kidney = (annotations_data[j,:,:] == 1).astype(int)*255
        tumor = (annotations_data[j,:,:] == 2).astype(int)*255

        scipy.misc.imsave('{}{}_{}.jpg'.format(images_path, case_folders[i], str(j)), images_data[j,:,:])
        scipy.misc.imsave('{}{}_{}_background.png'.format(annotations_path, case_folders[i], str(j)), background.astype(np.uint8))
        scipy.misc.imsave('{}{}_{}_kidney.png'.format(annotations_path, case_folders[i], str(j)), kidney.astype(np.uint8))
        scipy.misc.imsave('{}{}_{}_tumor.png'.format(annotations_path, case_folders[i], str(j)), tumor.astype(np.uint8))





