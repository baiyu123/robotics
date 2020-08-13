from simulator import *
import numpy as np
from scipy.stats import multivariate_normal
import math
import copy
# var = multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]])
# var.pdf([1,0])

class particle_class:
    pose = pose(0,0,0)
    landmark_list = []
    cov = None
    weight = 0
    # get landmark index, none if not found
    def find_landmark(self,id):
        for index in range(len(self.landmark_list)):
            if id == self.landmark_list[index].id:
                return index
        return None



class FastSlam:
    # particle
    particle_list = []
    x_min = -0.5
    x_max = 0.5
    y_min = -0.5
    y_max = 0.5
    theta_min = 0
    theta_max = 0
    particle_num = 50
    resample_num = 50
    delta_t = 0.1
    # velocity motion model
    alpha1 = 1
    alpha2 = 1
    alpha3 = 1
    alpha4 = 1
    alpha5 = 2
    alpha6 = 2
    #sensor
    Q_sensor = np.array([[0.05, 0],
                         [0, 0.1]])

    def __init__(self):
        # scatter particles
        for i in range(self.particle_num):
            x = np.random.uniform(self.x_min, self.x_max)
            y = np.random.uniform(self.y_min, self.y_max)
            theta = np.random.uniform(self.theta_min, self.theta_max)
            theta = 0
            particle_pose = pose(x,y,theta)
            particle = particle_class()
            particle.pose = particle_pose
            particle.weight = 0
            particle.cov = np.array([[0.5, 0, 0],
                                     [0, 0.5, 0],
                                     [0, 0, 0.1]])
            particle.landmark_list = []
            self.particle_list.append(particle)
        pass

    
    def get_measurement_predict(self, robot_pose, land_pose):
        r = np.sqrt((land_pose.x-robot_pose.x)**2 + (land_pose.y - robot_pose.y)**2)
        theta = np.arctan2(land_pose.y-robot_pose.y, land_pose.x - robot_pose.x) - robot_pose.theta
        theta %= 2*np.pi
        h = np.transpose(np.array([r, theta]))
        return h
    
    def get_landmark_mean(self, robot_pose, sensor):
        land_x = pose.x + sensor.r * np.cos(sensor.theta+robot_pose.theta)
        land_y = pose.y + sensor.r * np.sin(sensor.theta+robot_pose.theta)
        return land_x, land_y

    def get_H_jacobian(self, expect_pose, expect_landmark):
        q = (expect_landmark.x - expect_pose.x)**2 + (expect_landmark.y - expect_pose.y)**2
        H_1 = np.array([-(expect_landmark.x - expect_pose.x)/np.sqrt(q), -(expect_landmark.y - expect_pose.y)/np.sqrt(q)])
        H_2 = np.array([(expect_landmark.y - expect_pose.y)/q, -(expect_landmark.x - expect_pose.x)/q])
        H = np.array([H_1, H_2])
        return H

    def _calculateMeasurementJacobian(self, robotPose, featurePosition):
        rp = robotPose
        fp = featurePosition
        sqDist = (robotPose.x - featurePosition.x) ** 2 + (robotPose.y - featurePosition.y) ** 2
        dist = math.sqrt(sqDist)

        return np.array([
            [(rp.y - fp.y) / sqDist, (fp.x - rp.x) / sqDist],
            [(fp.x- rp.x) / dist, (fp.y - rp.y) / dist]
        ])

    def is_pos_def(self, x):
        return np.all(np.linalg.eigvals(x) > 0)
    
    def resample(self,num):
        # normalize
        sum = 0.0
        temp_list = []
        for particle in self.particle_list:
            if particle.weight > 0:
                sum += particle.weight
                temp_list.append(particle)
        weight_list = []
        for particle in temp_list:
            weight = particle.weight/sum
            weight_list.append(weight)
        new_particles = []
        for i in range(num):
            new_particle = np.random.choice(temp_list,p=weight_list)
            new_particles.append(copy.deepcopy(new_particle))
        self.particle_list = new_particles


    def update(self, sensor_data, command):
        for particle in self.particle_list:
            # motion update for particles
            p_pose = particle.pose
            v_bar = command.vel + np.random.normal(0,self.alpha1*command.vel**2+self.alpha2*command.angular_vel**2)
            w_bar = command.angular_vel + np.random.normal(0, self.alpha3*command.vel**2+self.alpha4*command.angular_vel**2)
            beta_bar = np.random.normal(0, self.alpha5*command.vel**2+self.alpha6*command.angular_vel**2)
            p_pose.x += -(v_bar/w_bar)*np.sin(p_pose.theta) + (v_bar/w_bar)*np.sin(p_pose.theta+w_bar*self.delta_t)
            p_pose.y += (v_bar/w_bar)*np.cos(p_pose.theta) - (v_bar/w_bar)*np.cos(p_pose.theta+w_bar*self.delta_t)
            p_pose.theta += (w_bar*self.delta_t + beta_bar*self.delta_t)%(2*np.pi)
            for data in sensor_data:
                index = particle.find_landmark(data.id)
                if index == None:
                    landmk = landmark(0,0)
                    landmk.id = data.id
                    landmk.x, landmk.y = self.get_landmark_mean(particle.pose, data)
                    H = self.get_H_jacobian(p_pose, landmk)
                    H2 = self._calculateMeasurementJacobian(p_pose, landmk)
                    H_inv = np.linalg.inv(H)
                    landmk.cov = np.dot(H_inv, np.dot(self.Q_sensor, np.transpose(H_inv)))
                    # if not self.is_pos_def(H_inv):
                    #     print('ha')
                    particle.landmark_list.append(landmk)
                    # eig = np.linalg.eig(landmk.cov)
                    # print(eig)
                else:
                    landmk = particle.landmark_list[index]
                    expect_z = self.get_measurement_predict(p_pose, landmk)
                    H = self.get_H_jacobian(p_pose, landmk)
                    Q_j = np.dot(np.dot(H, landmk.cov),np.transpose(H)) + self.Q_sensor
                    # EKF
                    K = np.dot(landmk.cov, np.transpose(np.dot(H, np.transpose(landmk.cov))))
                    mean_mat = np.array([landmk.x, landmk.y])
                    actual_z = np.array([data.r, data.theta])
                    mean_mat += np.dot(K, (actual_z - expect_z))
                    landmk.x = mean_mat[0]
                    landmk.y = mean_mat[1]
                    landmk.cov = np.dot((np.identity(2) - np.dot(K, H)), landmk.cov)
                    # weight
                    diff_z = actual_z-expect_z
                    
                    # if self.is_pos_def(Q_j):
                    var = multivariate_normal(mean=[0,0], cov=Q_j)
                    w1 = var.pdf(actual_z-expect_z)  
                    # else:
                        # w = 0
                    w = 1/np.sqrt(2*np.pi*Q_j ) * np.exp( np.transpose(-1/2 * np.dot(diff_z, np.dot(np.linalg.inv(Q_j ), diff_z))))
                    w = w[0][0]
                    # baixiao
                    # z_inov = (actual_z - expect_z).transpose()
                    # m_cov = Q_j
                    # w_b = 1.0 / math.sqrt(np.fabs(np.linalg.det(2 * math.pi * m_cov)))\
                    #     * np.exp(- 1.0 / 2.0 * np.dot(z_inov.transpose(), np.dot(np.linalg.inv(m_cov), z_inov)))
                    # print(w1-w)
                    w1 = min(1.0, w1)
                    landmk.weight = w1
            # sample particle
            particle_weight = 1
            for landmk in particle.landmark_list:
                particle_weight *= landmk.weight
                if math.isnan(particle_weight):
                    particle_weight = 0
            # print("weight: " + str(particle_weight))
            # print("theta: " + str(particle.pose.theta))
            particle.weight = particle_weight
        self.resample(self.resample_num)
    def all_pose(self):
        x = []
        y = []
        for particle in self.particle_list:
            x.append(particle.pose.x)
            y.append(particle.pose.y)
        return x, y
    def avg_pose(self):
        x = []
        y = []
        for particle in self.particle_list:
            x.append(particle.pose.x)
            y.append(particle.pose.y)
        x = sum(x)/len(x)
        y = sum(y)/len(y)
        return x, y
    def highest_pose(self):
        max = 0
        max_part = 0
        for particle in self.particle_list:
            if(particle.weight > max):
                max = particle.weight
                max_part = particle
        return max_part.pose.x, max_part.pose.y


# move_list = [[0.2, 0.1, ]]
sim = simulator()
slam = FastSlam()
sim.move(1, 0.1)
x = []
y = []
x_act = []
y_act = []
for i in range(20):
    print("calculate: " + str(i/50.0))
    sim.update()
    slam.update(sim.sense(), sim.get_command())

    x.append(slam.highest_pose()[0])
    y.append(slam.highest_pose()[1])
    x_act.append(sim.get_pose().x)
    y_act.append(sim.get_pose().y)


    # x, y = slam.all_pose()
    # act_pose = sim.get_pose()
    # x_act = act_pose.x
    # y_act = act_pose.y

    plt.plot(x,y,'ro')
    plt.plot(x_act, y_act, 'g^')
    plt.show()




# x, y = slam.all_pose()
# act_pose = sim.get_pose()
# x_act = act_pose.x
# y_act = act_pose.y

# plt.plot(x,y,'ro')
# plt.plot(x_act, y_act, 'g^')
# plt.show()










