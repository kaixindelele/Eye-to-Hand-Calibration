# Eye-to-Hand-Calibration

Eye-to-Hand Calibration，摄像机固定，与机器人基坐标系相对位置不变。且机器人末端在固定平面移动，即只需要求一个单应性矩阵的变换关系就行。

## 实验流程如下：

1. 手眼系统场景搭建：相机固定，机械臂带动针尖在固定平面移动。

2. 标定样本采集。包括摄像机图像采集，以及对应的机器人关节构型采集。--calibration_data_collected_main.py

3. 图像处理提取标定针尖，并计算针尖在机器人坐标系下坐标。记录好每个位置点的 针尖像素坐标、针尖世界坐标、末端坐标

4. 计算标定参数矩阵M--calibration_class.py

5. 计算重投影误差avg_e--calibration_class.py

## 标定实验的主要环境配置和使用到的工具有：

操作系统：Windows 7 64bit

图像处理工具：OpenCV-Python 3.4.* 
如果安装不上的话，版本是4.* 以上，用estimated2DAffine好像也行，没测试过。

机器人和摄像机：新松SCR5七自由度协作机械臂，海康工业相机MV-CA013-21UC

**其中calibration_class.py可以单独使用。只要有独立存在的标定点集即可。**
