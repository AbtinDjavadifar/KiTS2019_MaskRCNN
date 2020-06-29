# Kidney Tumor Detection Using MaskRCNN

This repository contains all the necessary files to train MaskRCNN model on KiTS (Kidney Tumor Segmentation Challenge) 2019 dataset. 

## Installing requirements:
`pip install requirements.txt`

## Preprocessing the data (creating images, masks, and annotations)

`python preprocess.py` 

## Training the model
Run `./train.sh`

## Additional resources
### Installing conda, tensorflow-gpu, and keras on ubuntu 18-04
https://medium.com/@naomi.fridman/install-conda-tensorflow-gpu-and-keras-on-ubuntu-18-04-1b403e740e25

### Some useful Repos:
https://github.com/neheller/kits19.git
https://github.com/facebookresearch/Detectron.git
https://github.com/facebookresearch/maskrcnn-benchmark.git
https://github.com/waspinator/pycococreator.git