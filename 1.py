import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
#This way it doesn't try to open a window un the GUI - works in python notebook


cam_mtx = np.array([
    [358.5, 0.0,   512.0],
    [0.0,   358.5, 256.0],
    [0.0,   0.0,   1.0],
    
])

colors = [(0,0,255), (0,255,0), (255,0,255), (0,255,255), (255,0,0)]
classNames = ["Car", "Truck", "Motorcycle", "Bicycle", "Pedestrian"]

'''
The key element of this dictionary is the semantic ID of a class you can use with
neural networks, while the value is the RGB color used in the images.
'''
semSegClasses = {  
     0: [0, 0, 0],         # None 
     1: [70, 70, 70],      # Buildings 
     2: [190, 153, 153],   # Fences 
     3: [72, 0, 90],       # Other 
     4: [220, 20, 60],     # Pedestrians 
     5: [153, 153, 153],   # Poles 
     6: [157, 234, 50],    # RoadLines 
     7: [128, 64, 128],    # Roads 
     8: [244, 35, 232],    # Sidewalks 
     9: [107, 142, 35],    # Vegetation 
     10: [0, 0, 255],      # Vehicles 
     11: [102, 102, 156],  # Walls 
     12: [220, 220, 0]     # TrafficSigns 
 } 

def drawBBs(BBs, img):
    H, W = img.shape[:2]
    print(BBs)
    for BB in BBs:
        u = BB[1]*W
        v = BB[2]*H
        w = BB[3]*W
        h = BB[4]*H
        c = int(BB[0])
        x_min = BB[5]
        x_max = BB[6]
        y_min = BB[5]
        y_max = BB[6]
        z_min = BB[5]
        z_max = BB[6]
        s = (int(u - w // 2), int(v - h // 2))
        e = (int(u + w // 2), int(v + h // 2))
        cv2.rectangle(img, s, e, colors[c], 1)
        tl = (s[0], s[1]+15)
        bl = (s[0], e[1]-5)
        cv2.putText(img,classNames[c],tl,cv2.FONT_HERSHEY_COMPLEX_SMALL,0.75,colors[c])
        #coords = "(%.2f, %.2f, %.2f, %.2f, %.2f, %.2f)" % (x_min, x_max, y_min, y_max, z_min, z_max)
        #cv2.putText(img,coords,bl,cv2.FONT_HERSHEY_COMPLEX_SMALL,0.65,colors[c])
    
    return img

# Read images
img = cv2.imread("carla_images/rgb/001/00.jpg")
depth = cv2.imread("carla_images/depth/000/00.png", -1)
seg = cv2.imread("carla_images/seg/000/00.png")
fs = cv2.imread("carla_images/seg/000/fs_00.png", -1)

# Read annotations
labels = np.loadtxt("carla_images/annotation/001/00_.txt")
pos = np.load("carla_images/annotation/000/00_.npy").reshape((4,4))

# Visualization
img = drawBBs(labels, img)
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
seg_rgb = cv2.cvtColor(seg,cv2.COLOR_BGR2RGB)

cv2.imshow('1',img_rgb)


cv2.waitKey(0)

