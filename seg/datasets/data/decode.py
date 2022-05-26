import os
import cv2
from PIL import Image
import numpy as np
imgs = os.listdir('PNGImages')

# imgs = [c[:-4] for c in imgs]

# with open('ImageSets/Segmentation/train.txt','w')as f:
#     for line in imgs:
#         f.write(line + '\n')
#
# with open('ImageSets/Segmentation/trainval.txt','w')as f:
#     for line in imgs:
#         f.write(line + '\n')

for im in imgs:
    img = cv2.imread('PNGImages/'+im)
    mk = cv2.imread('PedMasks/'+im[:-3]+'png')

    mk2 = Image.open('PedMasks/' + im[:-3] + 'png').convert('RGB')

    cv2.imshow('m',img)

    cv2.imshow('1',mk)
    cv2.imshow('2', np.array(mk2))
    cv2.waitKey(0)