# -*- encoding: utf-8 -*-
'''
@File    :  sausn.py
@Time    :  2020/10/04 22:24:26
@Author  :  Luo Yongle
1.获取当前末端位姿
2.for循环走九点
    3.获取末端位姿
    4.计算针尖位姿
    5.保存两个位姿
    6.保存图片
'''
from SCR_robot_task_space_class import Robot
from needle_detection import NeedleTip
import cv2
import csv
import os


robot = Robot(ip_address="192.168.98.66",)
needle_tip = NeedleTip(tip_point_pixel=1, visual_flag=False)


robot.reset()
obs = robot._get_obs()

# 单次进针，限定x, y, rz可动。
image_count = 0
data_list = []

calibration_files = "calibration_last"
try:
    os.mkdir(calibration_files)
    os.mkdir(calibration_files+'/origin')
    os.mkdir(calibration_files+'/detect')
except:
    pass

for x_index in range(6):
    for y_index in range(5):
        if 1 < y_index < 5:
            for rz_index in range(3):
                robot.reset()
                step_list = []

                robot.tip_step(dx=-10*x_index-20, 
                            dy=-10*y_index, 
                            drz=-2*rz_index)
                image = robot.camera.get_image_array()
                cv2.imwrite(calibration_files+'/origin/origin_image_index_'+str(image_count)+'.jpg', image)
                # 单次检测
                u, v = needle_tip.detection(image, 
                                            show_flag=True, 
                                            save_path=calibration_files+'/detect/dect_image_index_'+str(image_count)+'.jpg')
                obs = robot._get_obs()                                            
                eef_point = obs['terminal_pos']
                tip_point = obs['tip']         

                step_list.append(eef_point)
                step_list.append(tip_point)
                step_list.append([u, v])

                uv_right = input("n?:")
                if uv_right == 'n':
                    print("Discard the picture!")
                    print("saved image nums:", image_count)
                    break
                data_list.append(step_list)
                
                image_count += 1
                print("saved image nums:", image_count)
                


headers = ['eef_point','tip_point','uv']

with open(calibration_files+'/calibration_data.csv','w')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(data_list)



    

