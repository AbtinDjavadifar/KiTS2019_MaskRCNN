import os
import shutil
import random


dir = '/home/aeroclub/Abtin/train/kidneys_train2019/'
outputdir = '/home/aeroclub/Abtin/train/kidneys_val2019/'

files = [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]

val_amount = round(0.2*len(files))

for x in range(val_amount):

    file = random.choice(files)
    files.remove(file)
    shutil.move(os.path.join(dir, file), outputdir)


