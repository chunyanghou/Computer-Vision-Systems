import os
import random
import cv2
import numpy as np
from tqdm import tqdm
import shutil

folders = ['000','001','002','003','004','005','006','007']

imgs = []
labels = []

for fl in folders:
    fll = 'carla_images/rgb/' + fl
    m = os.listdir(fll)
    for mm in m:
        im = fl + '/' + mm
        imgs.append(im)
        labels.append(im.replace('rgb','annotation').replace('.jpg','_.txt'))

print(imgs)
num =len(imgs)
random.shuffle(imgs)
train_imgs = imgs[:int(0.7*num)]
test_imgs = imgs[int(0.7*num):]

# for im in train_imgs:
#     shutil.copy('carla_images/rgb/' +im,'2d-det/carla_image/train/images/'+im.replace('/','_'))
#     txt = 'carla_images/annotation/' + im.replace('.jpg','_.txt')
#     with open(txt,'r')as f:
#         fc = f.readlines()
#     with open('2d-det/carla_image/train/labels/' + im.replace('/','_').replace('.jpg','.txt'),'w')as tx:
#         for line in fc:
#             lin = line[:-1].split()[:5]
#             lin = ' '.join(lin) + '\n'
#             tx.write(lin)
#
# for im in test_imgs:
#     shutil.copy('carla_images/rgb/' +im,'2d-det/carla_image/test/images/'+im.replace('/','_'))
#     txt = 'carla_images/annotation/' + im.replace('.jpg','_.txt')
#     with open(txt,'r')as f:
#         fc = f.readlines()
#     with open('2d-det/carla_image/test/labels/' + im.replace('/','_').replace('.jpg','.txt'),'w')as tx:
#         for line in fc:
#             lin = line[:-1].split()[:5]
#             lin = ' '.join(lin) + '\n'
#             tx.write(lin)

for im in train_imgs:
    shutil.copy('carla_images/rgb/' +im,'seg/datasets/data/train/images/'+im.replace('/','_'))
    shutil.copy('carla_images/seg/' +im.replace('jpg','png'),'seg/datasets/data/train/masks/'+im.replace('/','_').replace('jpg','png'))


for im in test_imgs:
    shutil.copy('carla_images/rgb/' + im,'seg/datasets/data/test/images/'+im.replace('/','_'))
    shutil.copy('carla_images/seg/' + im.replace('jpg','png'),'seg/datasets/data/test/masks/'+im.replace('/','_').replace('jpg','png'))