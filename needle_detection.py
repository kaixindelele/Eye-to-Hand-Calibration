"""
1.实时获取相机rgb图片
2.灰度化
3.高斯模糊移除噪点
4.阈值化提取针体
5.极点检测提取针头
6.卡尔曼滤波获取平滑轨迹


# 设定机械臂初始状态；

尝试标定机器人和针尖位置；
鼠标交互；或者直接输入像素坐标uv；
空间上的指哪打哪；
尝试一下目标检测；

"""
import os
import math
import numpy as np
import cv2
import time
from PIL import Image
import matplotlib.pyplot as plt
import imutils
import sys
sys.path.append("D:\robot_code\luoyongle\calibration")
from hk_class import HHV


class NeedleTip:
    def __init__(self,
                 visual_flag=True,
                 tip_point_pixel=2, 
                 camera=None):        
        self.visual_flag = visual_flag
        self.tip_point_pixel = tip_point_pixel
        self.last_value = None
        if camera:
            self.cam = camera
        else:
            self.cam = HHV()
        # while 1:
        #     self.detection()

    def detection(self, image=None, show_flag=False, save_path=None):
        st = time.time()
        if image is not None:
            img = image
        else:
            img = self.cam.get_image_array()
        self.visual(img, "img")

        binary = self.threshold_demo(img, threshold=100,)
        self.visual(binary, "binary")

        roi = self.roi_select(binary, crop_value=30)
        if self.visual_flag:
            print("roi:", roi)
        (x0, y0), (x1, y1) = roi
        roi_img = binary[y0:y1, x0:x1].copy()
        self.visual(roi_img, "roi_img")
        tip_point = self.detection_tip(roi_img=roi_img)            
        
        # 二次精算
        box_size = 40
        second_x0 = x0 + tip_point[0] - box_size
        second_x1 = x0 + tip_point[0] + box_size
        second_y0 = y0 + tip_point[1] - box_size
        second_y1 = y0 + tip_point[1] + box_size

        second_roi_img = img[second_y0:second_y1, 
                             second_x0:second_x1].copy()
        self.visual(second_roi_img, "second_roi_img")
        # print("second_roi_img", second_roi_img.shape)

        # second_roi_binary = self.threshold_demo(second_roi_img, adaptive=True)
        second_roi_binary = self.threshold_demo(second_roi_img, threshold=20)
        
        tip_point = self.detection_tip(roi_img=second_roi_binary)
        # end
        print("st:", time.time() - st)

        height_crop = 100
        show_img = img[height_crop:, :, :].copy()
        roi_x, roi_y = tip_point
        x = second_x0 + roi_x
        y = second_y0 + roi_y - height_crop
        cv2.circle(show_img, (x, y), self.tip_point_pixel, (255, 0, 0), -1)
                                    
        # self.visual(show_img, "show_img")
        if show_flag:
            cv2.imshow("show_img", show_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_path is not None:
            save_img = np.array(show_img, dtype='uint8')
            cv2.imwrite(save_path, show_img)

        return (x, y)


    def visual(self, img, img_name='img', save_flag=False):
        if self.visual_flag:
            cv2.imshow(img_name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            if save_flag:
                cv2.write(img_name, img)
    
    def read_imgs_from_files(self, img_dir):
        img_list = []
        for img_name in os.listdir(img_dir):
            if 'bmp' in img_name or 'jpg' in img_name:                
                img_path = os.path.join(img_dir, img_name)                
                img = cv2.imread(img_path)                
                img_list.append(img)
        return img_list

    def threshold_demo(self, image, threshold=250, adaptive=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if adaptive:
            # 3 为Block size, 5为param1值
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,5)
            return binary
        else:
            ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)        
            return binary

    def find_array_boundary(self, arr, tip_lens=20):
        length = len(arr)
        # arr_s = reversed(arr[:length//2])
        arr_s = arr[:length//2]
        arr_s = arr_s[::-1]
        arr_b = arr[length//2:]
        for index, value in enumerate(arr_s):
            if value < 10 and (sum(arr_s[index:index+tip_lens]) < 200):
                x0 = length//2 - index
                break
        for index, value in enumerate(arr_b):
            if value < 10 and (sum(arr_s[index:index+tip_lens]) < 200):
                x1 = length//2 + index
                break
        return (x0, x1)

    def get_avg_boundary(self, tri_list):
        tri_list.sort(reverse=False)
        sorted_list = tri_list[:]
        dist1 = sorted_list[1] - sorted_list[0]
        dist2 = sorted_list[2] - sorted_list[1]
        if dist2 > dist1:
            avg = sum(sorted_list[:2])/2
        else:
            avg = sum(sorted_list[1:])/2
        return int(avg)
        
    def roi_select(self, binary, crop_value=20, tip_threshold=20):
        img = binary.copy()
        height, width = np.array(img).shape
        # 先找x0和x1
        x0_list, x1_list = [], []
        for i in range(3):
            arr = img[height//2-i*tip_threshold, :]
            x0, x1 = self.find_array_boundary(arr=arr)
            x0_list.append(x0)
            x1_list.append(x1)
            # print("x01:", x0, x1)
        x0 = self.get_avg_boundary(x0_list) + crop_value
        x1 = self.get_avg_boundary(x1_list) - crop_value * 3

        y0_list, y1_list = [], []
        for i in range(3):
            arr = img[:, width//2-i*tip_threshold]
            y0, y1 = self.find_array_boundary(arr=arr)
            y0_list.append(y0)
            y1_list.append(y1)
            # print("y01:", y0, y1)
        y0 = self.get_avg_boundary(y0_list) + crop_value
        y1 = self.get_avg_boundary(y1_list) - crop_value * 3        
        roi = ((min(x0, x1), min(y0, y1)), 
                (max(x0, x1), max(y0, y1)))

        return roi

    def detection_tip(self, roi_img):
        points_num=10
        level=0.1
        minDistance=20
        corners = cv2.goodFeaturesToTrack(roi_img,
                                        points_num,
                                        level,
                                        minDistance=minDistance
                                        )

        try:
            corners = np.int0(corners)
            corners = corners.reshape(corners.shape[0], 2)
            six_corners = corners[:, :]
            tip_point = np.array([six_corners[np.argmax(six_corners[:, 1]), 0],
                                np.max(six_corners[:, 1])])
            self.last_value = tip_point
            return tip_point
        except:
            return self.last_value
    

def main():
    
    needle_tip = NeedleTip()        


if __name__ == "__main__":
    main()