import numpy as np
import matplotlib.pyplot as plt
id_count = 0
class pose:
    x = 0
    y = 0
    theta = 0
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

class landmark:
    id = 0
    x = 0
    y = 0
    # for kalman
    cov = None
    weight = 1
    def __init__(self, new_x, new_y):
        global id_count
        self.x = new_x
        self.y = new_y
        self.id = id_count
        id_count += 1

class sensor_data:
    id = 0
    r = 0
    theta = 0
    def __init__(self, new_id, new_r, new_theta):
        self.id = new_id
        self.r = new_r
        self.theta = new_theta

class cmd:
    vel = 0
    angular_vel = 0

class simulator:
    # states
    delta_t = 0.1
    robot_pose = pose(0,0,0)
    landmark_list = []
    time_lapse = 0
    # sensor
    sensor_range = 10
    sensor_sigma = 0.1
    sensor_theta_sigma = 0.02
    # velocity motion model
    alpha1 = 1
    alpha2 = 1
    alpha3 = 1
    alpha4 = 1
    alpha5 = 2
    alpha6 = 2
    # control
    command = cmd()
    
    def __init__(self):
        for i in range(1,20):
            new_landmark = landmark(0, i/2.0)
            self.landmark_list.append(new_landmark)
            new_landmark2 = landmark(3, i/2.0)
            self.landmark_list.append(new_landmark2)
        self.robot_pose.x = 0
        self.robot_pose.y = 0

 

    # update the simulator
    def update(self):
        v_bar = self.command.vel + np.random.normal(0,self.alpha1*self.command.vel**2+self.alpha2*self.command.angular_vel**2)
        w_bar = self.command.angular_vel + np.random.normal(0, self.alpha3*self.command.vel**2+self.alpha4*self.command.angular_vel**2)
        beta_bar = np.random.normal(0, self.alpha5*self.command.vel**2+self.alpha6*self.command.angular_vel**2)
        self.robot_pose.x += -(v_bar/w_bar)*np.sin(self.robot_pose.theta) + (v_bar/w_bar)*np.sin(self.robot_pose.theta+w_bar*self.delta_t)
        self.robot_pose.y += (v_bar/w_bar)*np.cos(self.robot_pose.theta) - (v_bar/w_bar)*np.cos(self.robot_pose.theta+w_bar*self.delta_t)
        self.robot_pose.theta += w_bar*self.delta_t + beta_bar*self.delta_t
        self.time_lapse += self.delta_t
        

    def landmark_dist(self, curr_landmark):
        dist = (curr_landmark.x - self.robot_pose.x)**2 + (curr_landmark.y - self.robot_pose.y)**2
        dist = np.sqrt(dist)
        return dist   


    def sense(self):
        discovered_landmark = []
        for elem in self.landmark_list:
            dist = self.landmark_dist(elem)
            if dist <= self.sensor_range:
                dist += np.random.normal(0,self.sensor_sigma)
                vec_x = np.array([1,0])
                vec_landmark = np.array([elem.x - self.robot_pose.x, elem.y - self.robot_pose.y])
                theta = np.arccos(np.dot(vec_x,vec_landmark)/(np.linalg.norm(vec_x)*np.linalg.norm(vec_landmark)))
                cross = np.cross(vec_x, vec_landmark)
                if cross < 0:
                    theta = 2*np.pi - theta
                theta += np.random.normal(0,self.sensor_theta_sigma) - self.robot_pose.theta
                sen_data = sensor_data(elem.id, dist, theta)
                discovered_landmark.append(sen_data)
        return discovered_landmark
    
    def move(self, vel, angular_vel):
        self.command.vel = vel
        self.command.angular_vel = angular_vel
    
    def get_pose(self):
        return self.robot_pose

    def get_command(self):
        return self.command

def test():
    sim = simulator()
    sim.move(0.2, 0.0)
    x = []
    y = []
    landmarks_x = []
    landmarks_y = []
    for i in range(1,500):
        sim.update()
        pose = sim.get_pose()
        x.append(pose.x)
        y.append(pose.y)
        sensor_data = sim.sense()
        for data in sensor_data:
            land_x = pose.x + data.r * np.cos(data.theta+pose.theta)
            land_y = pose.y + data.r * np.sin(data.theta+pose.theta)
            landmarks_x.append(land_x)
            landmarks_y.append(land_y)
        # print("x:" + str(pose.x)+ " y:" + str(pose.y) + " theta:" + str(pose.theta))
        
    plt.plot(x,y,'ro')
    plt.plot(landmarks_x, landmarks_y, 'g^')
    plt.show()

# test()
# x = np.array([1, 6, 3, 4, 5, 6, 7, 3, 9, 10])
# y = np.array([0, 5, 5, 5, 5, 5, 5, 5, 5, 10])
# print(np.cov([x,y]))
# print(np.var(x))
# print(np.var(y))
