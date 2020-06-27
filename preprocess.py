from utils import convert_images_to_masks, data_splitter, convert_masks_to_COCO

kits_data = './kits19/data'
kits_train = './kits19/KiTS_coco/train2019/'
kits_val = './kits19/KiTS_coco/val2019/'
kits_test = './kits19/KiTS_coco/test2019/'
kits_masks = './kits19/KiTS_coco/masks/'
kits_annotations = './kits19/KiTS_coco/annotations/'

if __name__ == "__main__":

    convert_images_to_masks(kits_data, kits_train, kits_masks)
    data_splitter(kits_train, kits_val, kits_test)
    convert_masks_to_COCO(kits_val, kits_masks, kits_annotations)