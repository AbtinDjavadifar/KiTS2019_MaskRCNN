#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 19:27:29 2019

@author: abtin
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json                         
from skimage import measure                     
from shapely.geometry import Polygon, MultiPolygon 


data_path = '/home/abtin/kits19/dataselected/'
output_path = '/home/abtin/kits19/output/'
case_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
case_folders.sort()

# These ids will be automatically increased as we go
annotation_id = 1
image_id = 1
coco_annotations = []

def create_sub_masks(mask_image):

    sub_masks = {}
    sub_masks['1'] = (mask_image == 1).astype(float)
    sub_masks['2'] = (mask_image == 2).astype(float)

    return sub_masks


def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    if sub_mask.all() != 0:
        contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')
    
        segmentations = []
        polygons = []
        for contour in contours:
            # Flip from (row, col) representation to (x, y)
            # and subtract the padding pixel
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col, row)
    
            # Make a polygon and simplify it
            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=False)
            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)
    
        # Combine the polygons to calculate the bounding box and area
        multi_poly = MultiPolygon(polygons)
        x, y, max_x, max_y = multi_poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = (x, y, width, height)
        area = multi_poly.area
    
        annotation = {
            'segmentation': segmentations,
            'iscrowd': is_crowd,
            'image_id': image_id,
            'category_id': category_id,
            'id': annotation_id,
            'bbox': bbox,
            'area': area
            }   
    else:
        annotation = {
            'segmentation': 'None',
            'iscrowd': is_crowd,
            'image_id': image_id,
            'category_id': category_id,
            'id': annotation_id,
            'bbox': 'None',
            'area': 'None'
            }   
            

    return annotation

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

        kidney_id, tumor_id = [1, 2]
        category_ids = {
                                '1': kidney_id,
                                '2': tumor_id
                        }
        is_crowd = 0    

        sub_masks = create_sub_masks(mask_image)
        for sub_mask in sub_masks.items():
            sub_mask=sub_mask[1]
            category_id = category_ids
            annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
            coco_annotations.append(annotation)
            annotation_id += 1
        image_id += 1

with open('kidney_coco.json', 'w') as outfile:  
    json.dump(coco_annotations, outfile)




