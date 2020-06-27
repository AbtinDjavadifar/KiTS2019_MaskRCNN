#!/usr/bin/env bash
cd ./Mask_RCNN/samples/kidneys

python3 kidneys.py train --dataset=./kits19/KiTS_coco --model=coco
