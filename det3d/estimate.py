import pandas as pd
import numpy as np
import cv2
import math
import time
from pathlib import Path
import torch

from models.experimental import attempt_load
from utils.datasets import  LoadImages
from utils.general import  non_max_suppression, scale_coords, xyxy2xywh,  set_logging



@torch.no_grad()
class DistanceEstimation:
    def __init__(self):
        self.W = 1024
        self.H = 512
        self.excel_path = 'camera_parameters.xlsx'

    def camera_parameters(self, excel_path):
        df_intrinsic = pd.read_excel(excel_path, sheet_name='内参矩阵', header=None)
        df_p = pd.read_excel(excel_path, sheet_name='外参矩阵', header=None)

        print('Extrinsics Matrix：', df_p.values.shape)
        print('Intrinsics Matrix：', df_intrinsic.values.shape)

        return df_p.values, df_intrinsic.values

    def object_point_world_position(self, u, v, w, h, p, k):
        u1 = u
        v1 = v + h / 2
        print('Key point coordinates：', u1, v1)

        alpha = -(90 + 0) / (2 * math.pi)
        peta = 0
        gama = -90 / (2 * math.pi)

        fx = k[0, 0]
        fy = k[1, 1]
        H = 1
        angle_a = 0
        angle_b = math.atan((v1 - self.H / 2) / fy)
        angle_c = angle_b + angle_a
        print('angle_b', angle_b)

        depth = (H / np.sin(angle_c)) * math.cos(angle_b)
        print('depth', depth)

        k_inv = np.linalg.inv(k)
        p_inv = np.linalg.inv(p)
        # print(p_inv)
        point_c = np.array([u1, v1, 1])
        point_c = np.transpose(point_c)
        # print('point_c', point_c)
        # print('k_inv', k_inv)
        c_position = np.matmul(k_inv, depth * point_c)
        # print('c_position', c_position)
        c_position = np.append(c_position, 1)
        c_position = np.transpose(c_position)
        c_position = np.matmul(p_inv, c_position)
        d1 = np.array((c_position[0], c_position[1]), dtype=float)
        return d1

    def distance(self, kuang, xw=5, yw=0.1):
        print('=' * 50)
        print('Start distance measurement')

        p, k = self.camera_parameters(self.excel_path)
        if len(kuang):
            obj_position = []
            u, v, w, h = kuang[1] * self.W, kuang[2] * self.H, kuang[3] * self.W, kuang[4] * self.H
            print('Target box', u, v, w, h)
            d1 = self.object_point_world_position(u, v, w, h, p, k)
        distance = 0
        # print('距离', d1)
        if d1[0] <= 0:
            d1[:] = 0
        else:
            distance = math.sqrt(math.pow(d1[0], 2) + math.pow(d1[1], 2))

        return distance,[u,v,w,h], d1


    def Detect(self, weights='best.pt',
               source='data/images',  # file/dir/URL/glob, 0 for webcam
               imgsz=640,  # inference size (pixels)
               conf_thres=0.25,  # confidence threshold
               iou_thres=0.45,  # NMS IOU threshold
               max_det=1000,  # maximum detections per image
               device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
               save_crop=False,  # save cropped prediction boxes
               classes=None,  # filter by class: --class 0, or --class 0 2 3
               agnostic_nms=False,  # class-agnostic NMS
               augment=False,  # augmented inference
               project='inference/output',  # save results to project/name
               half=False,  # use FP16 half-precision inference
               ):



        # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        save_dir = Path(project)

        # Initialize
        set_logging()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        half &= device.type != 'cpu'  # half precision only supported on CUDA 仅在使用CUDA时采用半精度

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        # imgsz = check_img_size(imgsz, s=stride)  # check image size  测距不要缩放图片
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        # Set Dataloader
        vid_path, vid_writer = None, None

        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        for a in dataset:
            print(len(a))
            path, img, im0s,_, vid_cap = a
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process detections 检测过程
            for i, det in enumerate(pred):  # detections per image

                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path  p为inference/images/demo_distance.mp4
                save_path = str(save_dir / p.name)  # img.jpg  inference/output/demo_distance.mp4
                txt_path = str(save_dir / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt   inference/output/demo_distance_frame
                # print('txt', txt_path)
                s += '%gx%g ' % img.shape[2:]  # print string 图片形状 eg.640X480
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    classNames = ["Car", "Truck", "Motorcycle", "Bicycle", "Pedestrian"]
                    # Write results
                    distances = []
                    boxes = []
                    labels = []
                    for *xyxy, conf, cls in reversed(det):

                        c = int(cls)
                        label = classNames[c]
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                        kuang = [int(cls), xywh[0], xywh[1], xywh[2], xywh[3]]

                        # print(imc.shape)


                        distance, box,d = self.distance(kuang)
                        distances.append(distance)
                        boxes.append(box)
                        labels.append(label)
                    return distances,boxes,labels


def draw_box(img,boxes,ds,lbs):
    img = cv2.imread(img)

    for box,d,lb in zip(boxes,ds,lbs):
        box = [int(c) for c in box]

        x,y,w,h = box
        x = x - w//2
        y = y-h//2
        draw = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text = lb + ',dist:' + str(round(d,3))
        cv2.putText(draw, text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return draw

if __name__ == '__main__':

    print('Start target detection and monocular ranging!')
    img = '../carla_images/rgb/000/00.jpg'
    DE = DistanceEstimation()

    dist ,bbox,label = DE.Detect(source=img)

    r = draw_box(img,bbox,dist,label)

    cv2.imshow('1',r)
    cv2.waitKey(0)
#
