# -*- encoding: utf-8 -*-
'''
@File    :  sausn.py
@Time    :  2020/10/04 22:24:26
@Author  :  Luo Yongle
'''

import imp
from math import cos, sin, pi, sqrt
from re import T
import socket
import struct
import numpy as np
import time
from hk_class import HHV


class Robot:
    def __init__(self,
                 reward_type="tanh",
                 goal_reward=30.0,
                 image_size=224,
                 reward_shaping=True,
                 control_freq=10,
                 ip_address="192.168.98.87",
                 table_level=0.04,
                 object_height=210,
                 move_speed=35,
                 move_acc=15,
                 test_gripper=False,
                 ):
        """
            定义初始的参数，客户端，控制频率
            相机分辨率等
        """        
        self.goal_info = np.array([0.0, 0.0, 0.0])
        self.goal_reward = goal_reward
        self.reward_shaping = True
        self.reward_type = reward_type
        self.has_sensors = False
        self.ip_address = ip_address
        self.client_rec = None
        self.switch_time = 0
        self.client_ctrl = self.client_connect(ip_address, 2000)
        self.camera = HHV()
        self.collision_flag = False
        self.table_level = table_level
        assert self.table_level < 2
        assert self.table_level > 0.03
        self.action_step = 0
        self.out_2 = 0
        self.out_3 = 0
        self.object_height = object_height
        self.move_speed = move_speed
        self.move_acc = move_acc
        self.last_action = self.defalut_reset_xyz
        self.exception_flag = False
        self.collision_flag = False
        self.dpc = None
        self.gripper_and_lift_flag = False           
        self.goal_info_list = []

    @property
    def action_dim(self):
        return 4

    @property
    def action_bound(self):
        return 1

    @property
    def increased_action_bound(self):
        return 230  # 80mm

    @property
    def defalut_reset_xyz(self):
        return np.array([-256.0, -435.0, 304.0])

    @property
    def defalut_pose(self):
        return np.array([-90.01, -45.01, 88.9])     

    @property
    def defalut_backup_pose(self):
        return np.array([-90.01, -45.01, 90.1])     

    @property
    def reset_xyz_limited(self):
        return np.array([[-321.0, -256.0], 
                         [-488.0, -435.0],
                         [305.01, 305.01],
                         [-90.01, -90.01],
                         [-45.01, -45.01],                     
                         [85.01, 92.01],
                         ]
                         )

    @property
    def xyz_limited(self):
        return np.array([[-321.0, -256.0], 
                         [-488.0, -435.0],
                         [305.01, 305.01],
                         [-90.01, -90.01],
                         [-45.01, -45.01],                     
                         [85.01, 92.01],
                         ]
                         )
    
    def reset(self, ):  
        reset_pose = np.concatenate((self.defalut_reset_xyz, self.defalut_pose))     
        self.movej_pose(movej_pose=reset_pose)         
        obs = self._get_obs()
        self.last_obs = obs
        return obs

    def tip_step(self, dx, dy, drz, threshold=100):
        if abs(dx) > threshold:
            dx = threshold * dx / abs(dx)
        if abs(dy) > threshold:
            dy = threshold * dy / abs(dy)
        if abs(drz) > threshold:
            drz = threshold * drz / abs(drz)
        cur_action = self.last_action[:]
        cur_action[0] += dx
        cur_action[1] += dy
        cur_action[5] += drz
        self.movej_pose(movej_pose=cur_action)

    def _get_obs(self, action_list=[0, 0, 0, 0]):
        obs = dict()
        obs["gripper_flag"] = False
        obs["lift_flag"] = False
        obs["target_action"] = action_list
        obs["cube_pos"] = self.goal_info        
        robot_2001_status = []
        loop_time = time.time()
        while len(robot_2001_status) < 1:
            robot_2001_status = self.get_current_msg()
            if time.time() - loop_time > 30:
                print("get_status-action_time:", self.action_step)
                print("Here is Exception!")
                self.exception_flag = True
                break            
            dkr_pos = self.translate_2001_msg(robot_2001_status,
                                            392, 439, 'double')

            joints_angles = self.translate_2001_msg(robot_2001_status,
                                                    0, 55, 'double')
            
            joints_vel = self.translate_2001_msg(robot_2001_status,
                                                112, 167, 'double')
            # print("obs_joints_vel:", joints_vel)
            
            dkr_pos = self.translate_2001_msg(robot_2001_status,
                                            392, 439, 'double')
            terminal_position = dkr_pos
            # print("terminal_position:", terminal_position)
            tip = np.array(
                self.calcu_terminal_paw_pos(
                    terminal_position,
                    )) / 1000.0
        obs["tip"] = tip
        obs["gripper_pos"] = np.array(self.calcu_terminal_paw_pos(terminal_position)) / 1000.0

        obs["terminal_pos"] = np.array(terminal_position[:6]) / 1000.0
        obs["joints_angles"] = np.array(joints_angles) / 90.0
        # print("joints_angles:", joints_angles)
        obs["joints_vel"] = np.array(joints_vel)/90.0
        end_limit_flag = self.space_limit(terminal_position)

        table_collision_flag, tip = self.table_collision_detection(dkr_pos)
        obs["end_limit_flag"] = end_limit_flag
        obs["table_collision_flag"] = table_collision_flag

        robot_states = [
            obs["terminal_pos"],
            obs["joints_angles"],
            obs["gripper_pos"],
            np.array([int(obs["gripper_flag"])]),
            np.array([int(obs["lift_flag"])])            
        ]

        object_states = [
            obs["cube_pos"],
        ]
        obs["robot-state"] = np.concatenate(robot_states)
        obs["object-state"] = np.concatenate(object_states)
        return obs

    def calcu_terminal_paw_pos(self, 
                               terminal_position,
                               paw_length=393.5,):
        """
            parameter: 
                terminal_position: a list of 6 parameter of the robot's terminal
                paw_length: length of paw
            return:
                paw_pos: a [x, y, z] coordinate list of the paw on the terminal
        """
        terminal_pos = np.transpose(np.matrix(terminal_position[:3]))        
        y_offset = 0.0

        paw_pos_terminal = np.transpose(np.matrix([0, y_offset, paw_length]))
        cos_z, sin_z = cos(terminal_position[5] / 180 * pi), sin(terminal_position[5] / 180 * pi)
        cos_y, sin_y = cos(terminal_position[4] / 180 * pi), sin(terminal_position[4] / 180 * pi)
        cos_x, sin_x = cos(terminal_position[3] / 180 * pi), sin(terminal_position[3] / 180 * pi)
        rotate_z = np.matrix([  [cos_z, -sin_z, 0],
                                [sin_z,  cos_z, 0],
                                [0    ,      0, 1]])
        rotate_y = np.matrix([  [ cos_y, 0, sin_y],
                                [     0, 1,     0],
                                [-sin_y, 0, cos_y]])
        rotate_x = np.matrix([  [1,     0,      0],
                                [0, cos_x, -sin_x],
                                [0, sin_x,  cos_x]])
        paw_pos = rotate_z * rotate_y * rotate_x * paw_pos_terminal + terminal_pos
        paw_pos_list = paw_pos.tolist()
        return list([paw_pos_list[0][0], paw_pos_list[1][0], paw_pos_list[2][0]])

    def step(self, action, epoch=0, step=0):
        self.exp_epoch = epoch
        self.exp_step = step
        info = None
        self.collision_flag = False
        # obs = self._pre_action(action)
        try:
            obs = self._pre_action(action)
        except Exception as e:
            print("pre_action_exception:", e)
            print('文件', e.__traceback__.tb_frame.f_globals['__file__'])
            print('行号', e.__traceback__.tb_lineno)
            obs = self.last_obs
            self.exception_flag = True
            print("Had exception_flag==True!")
            self.email = EmailHandler(user="2392760986", 
                                  password="asnspluurhkseccg", 
                                  )
            self.email.send_error_to_myqq(subject="Wind Connect Error!",)
            exit()
        reward, done = self.reward(obs)
        self.last_obs = obs
        return obs, reward, done, info

    def _pre_action(self, action):
        """
        action:xyz and gripper        
        """
        # xyz (-1, 1) --> increased_action --> (-20, 20)mm
        self.gripper_try = False
        increased_action_list = action[:3]
        increased_action = increased_action_list * self.increased_action_bound        
        action_list = self.last_action + increased_action
        self.movej_pose(action_list)
        # gripper (-1,1) close if g<0 else open
        if action[-1] < 0:
            if self.gripper_and_lift_flag:
                gripper_flag = True
                lift_flag = True
            else:
                gripper_flag = False
                lift_flag = False
        else:
            last_z = self.last_action[-1]
            if last_z < self.object_height + 50:
                gripper_flag, lift_flag = self.ctrl_lift()
                self.gripper_try = True
            else:
                gripper_flag = False
                lift_flag = False
        
        # 判断是否接触过物块，如果接触过且没提起来，夹爪抬到初始位置，重新拍，更新goal_info
        if gripper_flag and lift_flag == False:
            last_action = self.last_action            
            self.movej_pose(self.defalut_reset_xyz,
                            direct_flag=True)
            self.goal_info = self.detection_model.get_cube_pose_from_net(reset_flag=False, right_flag=True)
            self.goal_info = self.goal_info / 1000.0
                               
            self.movej_pose(last_action,
                            direct_flag=True)
        # 如果夹住了，目标位置变为夹爪末端位置
        elif gripper_flag and lift_flag:
            self.goal_info = self.last_action
            self.goal_info[-1] -= 235.0
            self.goal_info = np.array(self.goal_info) / 1000.0            
            self.gripper_and_lift_flag = True
            # print("self.gripper_and_lift_flag:", self.gripper_and_lift_flag)

        obs = self._get_obs()
        # update lift and gripper flag!
        obs["gripper_flag"] = gripper_flag
        obs["lift_flag"] = lift_flag
        obs["cube_pos"] = self.goal_info
        robot_states = [
            obs["terminal_pos"],
            obs["joints_angles"],
            obs["gripper_pos"],
            np.array([int(obs["gripper_flag"])]),
            np.array([int(obs["lift_flag"])])            
        ]
        object_states = [
            obs["cube_pos"],
        ]
        obs["robot-state"] = np.concatenate(robot_states)
        obs["object-state"] = np.concatenate(object_states)
        return obs

    def movej_pose(self,
                   movej_pose, 
                   detect_collision_flag=False,
                   move_speed=25,
                   move_acc=15,
                   direct_flag=False,
                   ):
        # 限定末端位置在合适区间：
        if len(movej_pose)==3:
            movej_pose = [movej_pose[0], 
                          movej_pose[1],
                          movej_pose[2],
                          self.defalut_pose[0],
                          self.defalut_pose[1],
                          self.defalut_pose[2],
                          ]

        assert len(movej_pose)==6, '确保movej_pose为xyz,rxyz!'

        ctrl_range = self.xyz_limited
        reset_range = self.reset_xyz_limited
        if direct_flag:
            movej_pose = np.clip(np.array(movej_pose), 
                                 reset_range[:, 0], 
                                 reset_range[:, 1])
        else:   
            movej_pose = np.clip(np.array(movej_pose), 
                                 ctrl_range[:, 0], 
                                 ctrl_range[:, 1])
        # print("movej_pose:", movej_pose)
        action_list = [a for a in movej_pose]
        self.last_action = action_list[:]
        action_str = "["
        for act in action_list:
            action_str += str(int(act)) + ", "
        action_str = action_str[:-2]
        action_str += "]"        
        applied_string = "movej_pose(" + action_str + ", "+str(int(move_speed))+", "+str(int(move_acc))+");"
        
        self.client_ctrl.send(applied_string.encode("utf-8"))
        msg = self.client_ctrl.recv(1024).decode("utf-8")

        if msg == "Script start\n":
            action_start = time.time()
            stop_time = 0
            loop_time = 0
            self.client_ctrl.setblocking(False)
            while True:
                if self.dpc is not None:
                    self.dpc.step(epoch=self.exp_epoch, step=self.exp_step)
                tem_time = time.time()
                loop_time += 1
                self.client_ctrl.send("status".encode("utf-8"))
                self.client_ctrl.setblocking(True)
                status_msg = self.client_ctrl.recv(1024)
                origin_status_msg = status_msg.decode("ISO-8859-1")
                start_msg = status_msg.decode('utf-8')[:14]
                finish_msg = status_msg.decode('utf-8')[-14:]
                status_msg = origin_status_msg[:origin_status_msg.find('\n')]
                status_list = status_msg.split(';')
                terminal_position = status_list
                self.client_ctrl.setblocking(False)
                if len(origin_status_msg) > 110:
                    if "Script finish" in start_msg:
                        self.client_ctrl.setblocking(True)
                        break
                    if "Script finish" in finish_msg:
                        self.client_ctrl.setblocking(True)
                        break
                elif "Script finish" in status_msg:
                    self.client_ctrl.setblocking(True)
                    last_status_msg = self.client_ctrl.recv(1024)
                    last_status_msg = last_status_msg.decode("ISO-8859-1")
                    last_status_msg = last_status_msg[:status_msg.find('\n')]
                    break

                terminal_position = [float(x) for x in terminal_position]
                if detect_collision_flag:
                    collision_flag, tip = self.table_collision_detection(terminal_position)
                    if collision_flag:
                        if stop_time == 0:
                            self.client_ctrl.setblocking(True)
                            self.client_ctrl.send("stop".encode("utf-8"))
                            msg = self.client_ctrl.recv(1024).decode("utf-8")
                            if "Already stopped" in msg or msg == "Script stop\n":
                                stop_time = 1
                                self.client_ctrl.setblocking(True)
                                self.collision_flag = True
                                self.client_ctrl.close()
                                self.client_rec.close()
                                print("simple_collision_info:", end="\t")
                                break

                if time.time() - action_start > 15:
                    print("Action_error!" * 7)
                    print("last_msg:", status_list,
                          "loop_time:", loop_time,
                          )
                    
                    self.exception_flag = True
                    self.client_ctrl.setblocking(True)
                    self.email = EmailHandler(user="2392760986", 
                                              password="asnspluurhkseccg", 
                                              )
                    self.email.send_error_to_myqq(subject="Action Error!",)
                    exit()  
                    break

        self.client_ctrl.setblocking(True)
        self.action_step += 1

    def reset_from_zero(self,):
        reset_string = "movej([0, -30, 0, -58.80, 0, -91, 0], 30, 100, -1);"
        self.client_ctrl.send(reset_string.encode("utf-8"))
        while True:
            msg = self.client_ctrl.recv(1024)
            temp = msg.decode("utf-8")
            if msg.decode('utf-8')[-14:] == "Script finish\n":
                print("end_reset_from_zero")
                break

    def reward(self, obs, action=None):
        if self.reward_type == "tanh":
            return self.tanh_reward_function(obs, action)
        elif self.reward_type == "reletive":
            return self.reletive_reward_function(obs, action)
        elif self.reward_type == "l2":
            return self.negative_l2_reward_function(obs, action)

    def tanh_reward_function(self, obs, action=None):
        reward = 0.
        gripper_num = 0
        done_flag = False
        gripper_flag = obs["gripper_flag"]
        lift_flag = obs["lift_flag"]
        # use a shaping reward
        if self.reward_shaping:
            # reaching reward
            cube_pos = np.array(obs["cube_pos"], dtype=np.float64)
            # 将目标设为物块的正上方50厘米！            
            # cube center z is 31mm, 探索时最低是240-165=75mm,即需要将中心值提高50mm，才有可能拿到最高值1
            cube_pos[-1] += 0.05
            # gripper_site_pos = np.array(obs["gripper_pos"], dtype=np.float64)            
            gripper_site_pos = np.array(obs["gripper_pos"][:3], dtype=np.float64)

            dist = gripper_site_pos - cube_pos
            current_dist = np.linalg.norm(dist)
            reaching_reward = 1 - np.tanh(10 * current_dist)
            if gripper_flag:
                reward += 0.25            
            if lift_flag:
                reward += 1.0
                done_flag = True
            # if self.gripper_try and gripper_flag == False:
            #     reward -= 0.1
            reward += reaching_reward
        else:
            if gripper_flag and lift_flag:
                reward = 2.25
                done_flag = True
            else:
                reward = 0.0

        return reward, done_flag

    def client_connect(self, ip_address, port):
        """
            connnet to the robot
            ip_address: read the IP from the PC of robot
            port: 2000(control)   2001(current status)
        """
        client = socket.socket()
        client.connect((ip_address, port))
        return client

    def get_current_msg(self, msg_length=952):
        """
            get msg from port(2001)
            msg_length: 951+1 bytes----new version from page47
            Warning: if close time beyond 30000 with high frequency, may be rejected by robot server!
        """
        self.client_rec = self.client_connect(self.ip_address, 2001)
        status_msg = self.client_rec.recv(msg_length)
        self.client_rec.close()
        self.switch_time += 1
        return status_msg

    def translate_2001_msg(self,
                           msg,
                           start_byte,
                           end_byte,
                           msg_type):
        """
            translate msg of port(2001)
            parameters:
                msg: 856 bytes msg from port(2001)
                start_byte: 0 - 855 (int)
                end_byte: 0 - 855 (int)
                msg_type: 'double', 'char', 'bool'
                ###############################
                end_byte >= start_byte
                ###############################
            return:
                a list of translated msg
        """
        translate_msg = []
        if msg_type == 'double':
            while start_byte < end_byte:
                translate_msg.append(float('%.3f' % struct.unpack \
                    ('d', msg[start_byte: start_byte + 8])[0]))
                start_byte += 8
        elif msg_type == 'bool':
            while start_byte <= end_byte:
                translate_msg.append(struct.unpack('?', msg[start_byte: start_byte + 1])[0])
                start_byte += 1
        elif msg_type == 'char':
            while start_byte <= end_byte:
                translate_msg.append(struct.unpack('b', msg[start_byte: start_byte + 1])[0])
                start_byte += 1
        else:
            raise Exception("msg_type error!" + '\n' + "Not 'double', 'bool' or 'char'.")
        return translate_msg

    def table_collision_detection(self, terminal_position):
        collision_flag = False
        terminal_position = [float(x) for x in terminal_position]

        tip = np.array(
            self.calcu_terminal_paw_pos(
                terminal_position,
                )) / 1000.0
        return collision_flag, tip

    def space_limit(self, terminal_position, table_height=20):
        """
            set the space limit for the robot
            parameter: a list of 6 parameter of the robot's terminal
            return: True for the position is out of sapce limit and False for not
        """
        paw_length = 165
        paw_z = terminal_position[2] + paw_length * cos(terminal_position[3] / 180 * pi) * \
                cos(terminal_position[4] / 180 * pi)
        if paw_z < table_height:
            return True
        return False

    def move_8p_for_limited(self,):
        xy_limited = self.xyz_limited[:2]    
        z_limited = self.xyz_limited[2]  
        for z in z_limited:
            for x in xy_limited[0]:
                for y in xy_limited[1]:                
                    p = np.array([x, y, z])
                    self.movej_pose(movej_pose=p,
                                    direct_flag=True)
                

if __name__ == "__main__":
    robot = Robot(ip_address="192.168.99.87",
                  object_height=213,                  
                  test_gripper=False,
                  move_speed=45,
                  move_acc=15,
                  )
    # robot.env_reset()
    # robot.move_8p_for_limited()
    for i in range(6):
        print("i:", i)
        robot.env_reset()
        # robot.env_reset()

    