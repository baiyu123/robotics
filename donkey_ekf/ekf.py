import numpy as np

class calculated_landmark:
    def __init__(self, p, z, h):
        self.expected_z = z
        self.psi = p
        self.H = h

class donkey_ekf:
    
    def __init__(self):
        self.lr = 0.15
        self.L = 0.3
        self.dt = 0.1
        # motion noise
        self.Rt = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0.1]])
        # sensor noise
        self.range_var = 1
        self.angle_var = 0.1                                 
        self.Qt = np.array([[self.range_var, 0],
                            [0, self.angle_var]])



    def find_closest_feature(self, landmark_psi_exp_z, z):
        min_val = 999999
        min_index = 0
        for i in range(len(landmark_psi_exp_z)):
            val = np.transpose(z-landmark_psi_exp_z[i].expected_z)@np.linalg.inv(landmark_psi_exp_z[i].psi)@(z-landmark_psi_exp_z[i].expected_z)
            if val < min_val:
                min_val = val
                min_index = i
        return min_index

    def motion_update(self, miu, sigma, delta, v):
        beta = np.arctan(self.lr*np.tan(delta)/self.L)
        theta = miu[2]
        A = np.array([v*np.cos(theta+beta),
                      v*np.sin(theta+beta),
                      v*np.cos(beta)*np.tan(delta)/self.L])
        miu_bar = miu + A*self.dt
        G = np.array([[1, 0, -v*np.sin(theta+beta)],
                      [0, 1, v*np.cos(theta+beta)],
                      [0, 0, 1]])

        sigma_bar = G@sigma@np.transpose(G) + self.Rt
        return miu_bar, sigma_bar

    def expect_measure(self, land_mark_list, miu_bar, sigma_bar, Qt):
        landmark_psi_exp_z_H = [] # covariance and expected measurement of each landmark
        #  calculate measurement's covariance and expected measurement from map
        for elem in land_mark_list:
            landmark_x = elem[0]
            landmark_y = elem[1]
            delta_k = np.array([landmark_x-miu_bar[0], landmark_y-miu_bar[1]])
            q_k = np.transpose(delta_k)@delta_k
            expected_z = np.array([np.sqrt(q_k), np.arctan2(delta_k[1],delta_k[0]) - miu_bar[2]])
            H = (1/q_k)*np.array([[np.sqrt(q_k)*delta_k[0], -np.sqrt(q_k)*delta_k[1], 0],
                                  [delta_k[1], delta_k[0], -1]])
            psi_k = H@sigma_bar@np.transpose(H) + Qt
            land = calculated_landmark(psi_k, expected_z, H)
            landmark_psi_exp_z_H.append(land)
        return landmark_psi_exp_z_H

    def calculate_karman_gain(self, z_list, landmark_psi_exp_z, sigma_bar):
        # find the  measurement to corresponding landmark from map
        karman_gain_list = []
        measure_to_landmark_dic = {}
        for i in range(len(z_list)):
            z = z_list[i]
            index = self.find_closest_feature(landmark_psi_exp_z, z)
            H = landmark_psi_exp_z[index].H
            K = sigma_bar@H@np.linalg.inv(landmark_psi_exp_z[index].psi)
            measure_to_landmark_dic[i] = index
            karman_gain_list.append(K)
        return karman_gain_list, measure_to_landmark_dic

    def measurement_update(self, karman_gain_list, measure_to_landmark_dic, miu_bar, sigma_bar, z_list, H, landmark_psi_exp_z):
        miu_sum = np.array([[0],
                            [0],
                            [0]])
        sigma_sum = np.array([[0,0,0],
                              [0,0,0]])
        for z_index, index in measure_to_landmark_dic.items():
            miu_sum += karman_gain_list[z_index]*(z_list[z_index]-landmark_psi_exp_z[index].expected_z)
            sigma_sum += karman_gain_list[z_index]*H
        miu = miu_bar + miu_sum
        sigma = (np.identity - sigma_sum)@sigma_bar
        return miu, sigma

    def ekf_step(self, miu, sigma, delta, v, z_list, land_mark_list):
        miu_bar, sigma_bar = self.motion_update(miu, sigma,delta, v)
        landmark_psi_exp_z_H= self.expect_measure(land_mark_list, miu_bar, sigma_bar, self.Qt)
        karman_gain_list, measure_to_landmark_dic = self.calculate_karman_gain(z_list, landmark_psi_exp_z_H, sigma_bar)
        miu, sigma = self.measurement_update(karman_gain_list, measure_to_landmark_dic, miu_bar, sigma_bar, z_list, H, landmark_psi_exp_z)
        return miu, sigma


