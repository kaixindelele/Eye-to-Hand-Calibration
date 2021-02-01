# Eye-to-Hand-Calibration
Eye-to-Hand Calibration，摄像机固定，与机器人基坐标系相对位置不变。且机器人末端在固定平面移动，即只需要求一个单应性矩阵的变换关系就行。

实验流程如下：

1. 手眼系统场景搭建。
2. 标定样本采集。包括摄像机图像采集，以及对应的机器人关节构型采集。--calibration_data_collected_main.py
3. 图像处理提取标定针尖，并计算针尖在机器人坐标系下坐标。
4. 计算标定参数矩阵M--calibration_class.py
仿真实验的主要环境配置和使用到的工具有：

操作系统：Windows 7 64bit
图像处理工具：OpenCV
机器人和摄像机：新松SCR5七自由度协作机械臂，海康工业相机MV-CA013-21UC
虚拟棋盘图编辑：AC3D，用于生成带贴图的棋盘标定板，加载到手眼仿真场景中
