
import os
from os.path import join
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import gc
pixs_per_meter = 5
pixs_per_meter_heignt = 10
length, width = 512, 512
height = int(2*pixs_per_meter_heignt)
lidar_height = 2.5

class Location():
    def __init__(self, x=0., y=0., z=0.):
        self.x = x
        self.y = y
        self.z = z

class Rotation():
    def __init__(self, pitch=0., yaw=0., roll=0.):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

class Pose():
    def __init__(self):
        self.location = Location()
        self.rotation = Rotation()

class RobotInfo():
    def __init__(self):
        self.time_stamp = 0.
        self.pose = Pose()
        self.vel = Location()
        self.angular_velocity = Location()

class LidarDataset(Dataset):
    def __init__(self, dataset_dir, data_dirs=None):
        self.dataset_dir = dataset_dir
        self.data_dirs = data_dirs
        self.data_dict, self.pose_dict = self.find_data(self.dataset_dir)
        self.MAX_TRAJECTORY_TIME = 3.0
        self.IMG_HEIGHT = 200
        self.IMG_WIDTH = 400
        self.MAX_DIST = 25
        self.MAX_SPEED = 10
        self.FRAME_THRESHOLD = 100
        self.POINT_NUM = 16
        nav_transforms = [
            transforms.Resize((self.IMG_HEIGHT, self.IMG_WIDTH), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
        self.nav_transforms = transforms.Compose(nav_transforms)
        
    def find_data(self, dataset_dir):
        data_dict = {}
        pose_dict = {}
        data_list = os.listdir(dataset_dir)
        for sub_dir in data_list:
            if self.data_dirs is not None:
                if sub_dir in self.data_dirs:
                    pass
                else:
                    continue
            data_dir = dataset_dir+'/'+sub_dir
            with open(data_dir+'/pose.txt', 'r') as file:
                lines = file.readlines()
                pose_dict[sub_dir] = lines
                data_dict[sub_dir] = len(lines)
        return data_dict, pose_dict

    def get_lidar_data(self, key, index):
        assert key in self.data_dict.keys()
        assert index < self.data_dict[key]
        lidar_data = np.load(self.dataset_dir+'/'+key+'/pcd/'+str(index)+'.npy')
        return lidar_data

    def get_nav(self, key, index):
        assert key in self.data_dict.keys()
        assert index < self.data_dict[key]
        img = Image.open(self.dataset_dir+'/'+key+'/nav/'+str(index)+'.png').convert("L")
        img = self.nav_transforms(img)
        return img

    def get_vel(self, key, index):
        vx, vy = self.parse_vel(self.pose_dict[key][index])
        return vx, vy

    def get_trajectory(self, key, index):
        assert key in self.data_dict.keys()
        assert index < self.data_dict[key]
        time_stamp_list = []
        x_list = []
        y_list = []
        t0, x0, y0, yaw0 = self.parse_2d_pose(self.pose_dict[key][index])

        for i in range(self.FRAME_THRESHOLD):
            line = self.pose_dict[key][index + i]
            time_stamp, x, y, _ = self.parse_2d_pose(line)
            local_x, local_y = self.tf_pose(x, y, x0, y0, yaw0)
            time_stamp_list.append(time_stamp - t0)
            x_list.append(local_x)
            y_list.append(local_y)
            if time_stamp - t0 > self.MAX_TRAJECTORY_TIME:
                break

        return np.array(x_list).astype(np.float32), np.array(y_list).astype(np.float32), np.array(time_stamp_list).astype(np.float32)

    def parse_2d_pose(self, line):
        sp_line = line.split()
        time_stamp = float(sp_line[1])
        x = float(sp_line[2])
        y = float(sp_line[3])
        yaw = np.deg2rad(float(sp_line[6]))
        return time_stamp, x, y, yaw

    def tf_pose(self, x_t, y_t, x_0, y_0, yaw):
        dx = x_t - x_0
        dy = y_t - y_0
        x = np.cos(yaw)*dx + np.sin(yaw)*dy
        y = np.cos(yaw)*dy - np.sin(yaw)*dx
        return x, y

    def parse_vel(self, line):
        sp_line = line.split()
        yaw = np.deg2rad(float(sp_line[6]))
        vx_ = float(sp_line[8])
        vy_ = float(sp_line[9])
        vx = np.cos(yaw)*vx_ + np.sin(yaw)*vy_
        vy = np.cos(yaw)*vy_ - np.sin(yaw)*vx_
        return vx, vy

    def parse_pose(self, line):
        sp_line = line.split()
        robot_info = RobotInfo()
        robot_info.time_stamp = float(sp_line[1])
        robot_info.pose.location.x = float(sp_line[2])
        robot_info.pose.location.y = float(sp_line[3])
        robot_info.pose.location.z = float(sp_line[4])
        robot_info.pose.rotation.pitch = float(sp_line[5])
        robot_info.pose.rotation.yaw = float(sp_line[6])
        robot_info.pose.rotation.roll = float(sp_line[7])
        robot_info.vel.x = float(sp_line[8])
        robot_info.vel.y = float(sp_line[9])
        robot_info.vel.z = float(sp_line[10])
        robot_info.angular_velocity.x = float(sp_line[11])
        robot_info.angular_velocity.y = float(sp_line[12])
        robot_info.angular_velocity.z = float(sp_line[13])
        return robot_info

    def pcd2voxel(self, point_cloud):
        voxel = np.zeros((height, length, width), np.float32)
        voxel.fill(0.)
        
        u = (-point_cloud[0]*pixs_per_meter+length//2).astype(int)
        v = (-point_cloud[1]*pixs_per_meter+width//2).astype(int)
        p = ((point_cloud[2]+lidar_height)*pixs_per_meter_heignt).astype(int)
        
        mask = np.where((u >= 0)&(u < length)&(v >= 0)&(v < width)&(p >= 0)&(p < height))[0]
        u = u[mask]
        v = v[mask]
        p = p[mask]
        voxel[p,u,v] = 1.
        return voxel

    def __getitem__(self, item_index):
        key = random.choice(list(self.data_dict.keys()))
        index = random.randint(self.FRAME_THRESHOLD, self.data_dict[key]-self.FRAME_THRESHOLD)

        _lidar_data = self.get_lidar_data(key, index)
        lidar_data = self.pcd2voxel(_lidar_data)
        del _lidar_data; gc.collect()

        nav = self.get_nav(key, index)
        x, y, t = self.get_trajectory(key, index)
        vx0, vy0 = self.get_vel(key, index)
        
        mask = random.sample(list(range(len(x)))[1:-1], self.POINT_NUM-1)
        mask.append(0)
        mask.sort()

        x = x[mask]
        y = y[mask]
        t = t[mask]

        xy = torch.FloatTensor([x/self.MAX_DIST, y/self.MAX_DIST]).T
        v0 = torch.FloatTensor(np.array([vx0/self.MAX_SPEED, vy0/self.MAX_SPEED]).astype(np.float32)).T
        v0_array = torch.FloatTensor(np.array([vx0/self.MAX_SPEED, vy0/self.MAX_SPEED]*self.POINT_NUM).astype(np.float32)).T
        t = torch.FloatTensor([t/self.MAX_TRAJECTORY_TIME]).T

        return {
            'lidar': lidar_data,
            'nav': nav,
            'xy': xy,
            'v0': v0,
            'v0_array': v0_array,
            't': t,
            }

    def __len__(self):
        return 9999999999


if __name__ == '__main__':
    dataset = LidarDataset(dataset_dir='../LidarDataset')