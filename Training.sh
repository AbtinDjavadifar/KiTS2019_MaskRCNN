#!/usr/bin/env bash
cd /home/aeroclub/PycharmProjects/Kidneys_MaskRCNN/Mask_RCNN/samples/kidneys

python3 kidneys.py train --dataset=/home/aeroclub/Abtin/KiTS_coco --model=coco
