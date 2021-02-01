import cv2
import pandas as pd
import numpy as np


class Calibration:
    def __init__(self, 
                 csv_path='',
                 scale=1000.0,
                 train_num=33,
                 ):
        self.csv_path = 'calibration_last/calibration_data.csv'
        # 不放大到同一个数量级，会受到浮点数影响
        self.scale = scale
        self.train_num = train_num
        self.init_data_from_csv()
        self.uv2xy_m = self.get_m(origin_points_set=self.points_camera,
                                  target_points_set=self.points_robot,
                                  )
        print("uv2xy-m:", self.uv2xy_m)
        self.uv2xy_m_avg_e = self.reproject(origin_points_set=self.points_camera_test,
                                            target_points_set=self.points_robot_test,
                                            m=self.uv2xy_m)

        self.uv2xy_mt = self.get_mt(self.uv2xy_m)
        print("uv2xy-mt:", self.uv2xy_mt)
        self.uv2xy_mt_avg_e = self.reproject(origin_points_set=self.points_robot_test,
                                             target_points_set=self.points_camera_test,
                                             m=self.uv2xy_mt)

        # xy2uv
        self.xy2uv_m = self.get_m(origin_points_set=self.points_robot,
                                  target_points_set=self.points_camera,
                                  )
        print("xy2uv_m:", self.xy2uv_m)
        self.xy2uv_m_avg_e = self.reproject(origin_points_set=self.points_robot_test,
                                            target_points_set=self.points_camera_test,
                                            m=self.xy2uv_m)

        self.xy2uv_mt = self.get_mt(self.xy2uv_m)
        print("xy2uv_mt:", self.xy2uv_mt)
        self.xy2uv_mt_avg_e = self.reproject(origin_points_set=self.points_camera_test,
                                             target_points_set=self.points_robot_test,
                                             m=self.xy2uv_mt)
    
    def init_data_from_csv(self,):
        csv_frame = pd.read_csv(self.csv_path)
        all_data = np.array(csv_frame)
        uv = all_data[:, 2]
        points_camera = self.list_str2list(all_data[:, 2])
        points_robot = self.list_str2list(all_data[:, 1])[:, :2]
        self.points_robot = points_robot[:self.train_num] * self.scale
        self.points_camera = np.array(points_camera, dtype='float64')[:self.train_num]
        self.points_robot_test = points_robot[self.train_num:] * self.scale
        self.points_camera_test = np.array(points_camera, dtype='float64')[self.train_num:]        

    def get_m(self,
              origin_points_set,
              target_points_set):
        # 确保两个点集的数量级不要差距过大，否则会输出None,看到这个输出，我直接好家伙。
        # 明明回家前，还能输出一个好的转换矩阵，为什么一回家就报错？我错哪儿了...
        m = cv2.estimateRigidTransform(origin_points_set, 
                                       target_points_set,
                                       fullAffine=True)
        return m

    def get_mt(self, m):
        a, b, c = m[0,0], m[0,1], m[0,2]    
        d, e, f = m[1,0], m[1,1], m[1,2]
        denominator	= b*d-a*e
        mt = np.array([[-e,  b, -b*f+d*e],
                       [ d, -a,  a*f-c*d]])
        mt = mt / denominator
        return mt        

    def reproject(self, 
                  origin_points_set,
                  target_points_set,
                  m):
        error_list = []
        for index in range(len(origin_points_set)):
            p_origin = list(origin_points_set[index])
            p_origin.append(1)
            p_origin = np.array(p_origin)
            p_tar = target_points_set[index]    
            new_tar = np.dot(m, p_origin)
            error = np.linalg.norm(new_tar-p_tar[:2])            
            error_list.append(error)

        print("avg_e:", np.mean(np.array(error_list)))
        return np.mean(np.array(error_list))

    def list_str2list(self, list_str):
        temp_list = []
        for str in list_str:
            if ',' in str:
                new_list = eval(str)
                temp_list.append(new_list)
            else:
                k = str.replace('\n','').replace('    ',' ').replace('   ',' ').replace('  ',' ').replace(' ',',')
                if k[1] == ',':
                    k = k[0:1]+k[2:]
                new_list = eval(k)
                temp_list.append(new_list)
        return np.array(temp_list)    

    def uv2xy(self, uv):
        m = self.uv2xy_m
        p_origin = list(uv)
        p_origin.append(1)
        p_origin = np.array(p_origin)        
        xy = np.dot(m, p_origin)
        return xy

    def xy2uv(self, xy):
        m = self.xy2uv_m
        p_origin = list(xy)
        p_origin.append(1)
        p_origin = np.array(p_origin)        
        uv = np.dot(m, p_origin)
        return uv

def main():
    print('cv2.__version__', cv2.__version__)
    print("this code need 3.4.*")
    calib = Calibration(train_num=20)
    xy = calib.uv2xy((120, 20))
    print("xy:", xy)
    uv = calib.xy2uv((120, 200))
    print("uv:", uv)
    print(np.finfo(np.longdouble))
    # 测试不同拟合点数对测试集平均误差结果的影响：
    avg_e_list_list = [[], [], [], []]
    title_list = ['uv2xy_m_avg_e', 'uv2xy_mt_avg_e', 'xy2uv_m_avg_e', 'xy2uv_mt_avg_e']
    for index in range(20):
        calib = Calibration(train_num=index+3)
        avg_e_list_list[0].append(calib.uv2xy_m_avg_e)
        avg_e_list_list[1].append(calib.uv2xy_mt_avg_e)
        avg_e_list_list[2].append(calib.xy2uv_m_avg_e)
        avg_e_list_list[3].append(calib.xy2uv_mt_avg_e)
    
    import matplotlib.pyplot as plt
    for index, avg_e_list in enumerate(avg_e_list_list):
        plt.plot(avg_e_list)
        plt.title(title_list[index])
        plt.show()        


if __name__=='__main__':
    main()