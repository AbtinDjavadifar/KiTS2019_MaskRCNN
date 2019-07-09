import os
import shutil
import random


dir = '/home/abtin/kits19/KiTS_coco/train2019/'
valoutputdir = '/home/abtin/kits19/KiTS_coco/val2019/'
testoutputdir = '/home/abtin/kits19/KiTS_coco/test2019/'

files = [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]

val_amount = round(0.2*len(files))
test_amount = round(0.1*len(files))

for x in range(val_amount):

    file = random.choice(files)
    files.remove(file)
    shutil.move(os.path.join(dir, file), valoutputdir)

for x in range(test_amount):

    file = random.choice(files)
    files.remove(file)
    shutil.move(os.path.join(dir, file), testoutputdir)

