import os
import numpy as np
import nibabel as nib
import imageio
import shutil
import random
import datetime
import json
import re
import fnmatch
from PIL import Image
import pycococreatortools

def convert_images_to_masks(kits_data, kits_train, kits_masks):
    """

    Returns: Tumor and kidney masks

    """
    case_folders = [f for f in os.listdir(kits_data) if os.path.isdir(os.path.join(kits_data, f))]
    case_folders.sort()

    for i in range(len(case_folders)):

        case_path = os.path.join(kits_data, case_folders[i])
        case_images_path = os.path.join(case_path, 'imaging.nii.gz')
        case_annotations_path = os.path.join(case_path, 'segmentation.nii.gz')
        images = nib.load(case_images_path)
        annotations = nib.load(case_annotations_path)
        images_data = images.get_fdata()
        annotations_data = annotations.get_fdata()

        for j in range(np.shape(images_data)[0]):
            background = (annotations_data[j, :, :] == 0).astype(int) * 255
            kidney = (annotations_data[j, :, :] == 1).astype(int) * 255
            tumor = (annotations_data[j, :, :] == 2).astype(int) * 255

            imageio.imwrite('{}{}_{}.jpg'.format(kits_train, case_folders[i], str(j)), images_data[j, :, :])
            # scipy.misc.imsave('{}{}_{}_background.png'.format(masks_path, case_folders[i], str(j)), background.astype(np.uint8))
            imageio.imwrite('{}{}_{}_kidney.png'.format(kits_masks, case_folders[i], str(j)), kidney.astype(np.uint8))
            imageio.imwrite('{}{}_{}_tumor.png'.format(kits_masks, case_folders[i], str(j)), tumor.astype(np.uint8))

            print("Case number: {} Image Number: {}".format(i, j))


def data_splitter(kits_train, kits_val, kits_test):
    """

    Returns: train2019, val2019, and test2019 folders

    """

    files = [file for file in os.listdir(kits_train) if os.path.isfile(os.path.join(kits_train, file))]

    val_amount = round(0.2 * len(files))
    test_amount = round(0.1 * len(files))

    for x in range(val_amount):
        file = random.choice(files)
        files.remove(file)
        shutil.move(os.path.join(kits_train, file), kits_val)

    for x in range(test_amount):
        file = random.choice(files)
        files.remove(file)
        shutil.move(os.path.join(kits_train, file), kits_test)


def convert_masks_to_COCO(kits_val, kits_masks, kits_annotations):
    """

    Returns: Converted masks in COCO format

    """

    INFO = {
        "description": "KiTS Dataset",
        "url": "https://github.com/AbtinJ/Kidneys_MaskRCNN",
        "version": "1.0",
        "year": 2019,
        "contributor": "Abtin",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    CATEGORIES = [
        {
            'id': 1,
            'name': 'kidney',
            'supercategory': 'organ',
        },
        {
            'id': 2,
            'name': 'tumor',
            'supercategory': 'organ',
        },
    ]

    def filter_for_jpeg(root, files):
        file_types = ['*.jpeg', '*.jpg']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]

        return files

    def filter_for_annotations(root, files, image_filename):
        file_types = ['*.png']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
        file_name_prefix = basename_no_extension + '.*'
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]
        files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

        return files

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 0
    segmentation_id = 0

    # filter for jpeg images
    for root, _, files in os.walk(kits_val):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(kits_masks):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    print(annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                                             .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    # with open('{}/instances_train2019.json'.format(ANNOTATION_DIR), 'w') as output_json_file:
    #     json.dump(coco_output, output_json_file)

    with open('{}/instances_val2019.json'.format(kits_annotations), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)