#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '../'))

import simulator
simulator.load('/home/wang/CARLA_0.9.9.4')
import carla
sys.path.append('/home/wang/CARLA_0.9.9.4//PythonAPI/carla')
from agents.navigation.basic_agent import BasicAgent

from simulator import config, set_weather, add_vehicle
from simulator.sensor_manager import SensorManager
from utils.navigator_sim import get_map, get_nav, replan, close2dest
from utils import add_alpha_channel, debug
from controller import CapacController, getActorState

import os
import cv2
import time
import copy
import threading
import random
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import grad

from informer import Informer

def parse_traj(message, robot_id):
    global global_raw_trajectory, global_new_traj
    global_raw_trajectory = np.frombuffer(message, dtype=np.float32).reshape(-1, 2)
    global_new_traj = True

class Client(Informer):
    def send_nav(self, message):
        self.send(message, 'nav')
    
    def send_lidar(self, message):
        self.send(message, 'lidar')

    def traj_recv(self):
        self.recv('traj', parse_traj)

DEBUG_VISUALIZATION = True
TOTAL_DIST = 300
WRITE_LOG = True

MAX_SPEED = 30

global_view_img = None
global_nav = None
global_pcd = None
global_vel = 0
global_vehicle= None
global_plan_map = None
global_plan_time = None
state0 = None
global_trajectory = None
global_collision = False

global_raw_trajectory = None
global_new_traj = False

lidar_height = 2.5

pixs_per_meter = 5
pixs_per_meter_heignt = 10
length, width = 512, 512
height = int(2*pixs_per_meter_heignt)

ifm = Client(config = 'config.yaml')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="run-carla-01", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--checkpoint_interval', type=int, default=500, help='interval between model checkpoints')
parser.add_argument('--points_num', type=int, default=1, help='points number')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_speed', type=float, default=10., help='max speed')
parser.add_argument('--max_t', type=float, default=3., help='max time')
parser.add_argument('--dt', type=float, default=0.05, help='discretization minimum time interval')
opt = parser.parse_args()

IMG_HEIGHT = 200
IMG_WIDTH = 400
nav_transforms = [
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
]
nav_transforms = transforms.Compose(nav_transforms)

def collision_callback(data):
    global global_collision
    global_collision = True

def view_image_callback(data):
    global global_view_img, global_vehicle, global_plan_map, global_nav
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_view_img = array

def lidar_callback(data):
    global global_pcd, global_plan_time, global_vehicle, state0
    lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 3])
    mask = np.where((-lidar_data[:,2] < 5-lidar_height)&(-lidar_data[:,2] >= -lidar_height-0.1))[0]
    point_cloud = np.stack([-lidar_data[:,1][mask], -lidar_data[:,0][mask], -lidar_data[:,2][mask]])
    global_pcd = point_cloud

def pcd2voxel(point_cloud):
    u = (-point_cloud[0]*pixs_per_meter+length//2).astype(int)
    v = (-point_cloud[1]*pixs_per_meter+width//2).astype(int)
    p = ((point_cloud[2]+lidar_height)*pixs_per_meter_heignt).astype(int)
    
    mask = np.where((u >= 0)&(u < length)&(v >= 0)&(v < width)&(p >= 0)&(p < height))[0]
    u = u[mask]
    v = v[mask]
    p = p[mask]
    voxel_index = np.vstack([p, u, v]) # (3, 3384)
    return voxel_index

def visualize(input_img, nav=None):
    global global_vel
    img = copy.deepcopy(input_img)
    text = "speed: "+str(round(3.6*global_vel, 1))+' km/h'
    cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
    if nav is not None:
        new_nav = add_alpha_channel(nav)
        new_nav = cv2.flip(new_nav, 1)
        img[:nav.shape[0],-nav.shape[1]:] = new_nav

    cv2.imshow('Visualization', img)
    cv2.waitKey(5)


def wait_for_client(input_data):
    global ifm, global_new_traj, global_raw_trajectory
    # print('lidar and nav shape:', input_data['lidar'].shape, input_data['nav'].shape)
    # (20, 512, 512) (1, 200, 400)
    t1 = time.time()
    lidar = input_data['lidar'].tobytes()
    nav = input_data['nav'].tobytes()
    # 81216 320000

    ifm.send_lidar(lidar)
    ifm.send_nav(nav)
    print('send_lidar, send_nav')

    while not global_new_traj:
        # time.sleep(0.1)
        time.sleep(0.01)
        # ifm.send_lidar(lidar)
        # ifm.send_nav(nav)
        # print('send_lidar, send_nav')
        
    global_new_traj = False
    t2 = time.time()
    print('time used:', str(round(1000*(t2-t1), 1)), 'ms')
    return global_raw_trajectory


def get_trajectory(input_data):
    global global_plan_time, global_vehicle, state0
    global_plan_time = time.time()
    plan_time = global_plan_time
    transform = global_vehicle.get_transform()
    state0 = getActorState('odom', global_plan_time, global_vehicle)
    state0.x = transform.location.x
    state0.y = transform.location.y
    state0.z = transform.location.z
    state0.theta = np.deg2rad(transform.rotation.yaw)

    output = wait_for_client(input_data)
    print('get remote trajectory', output.shape)

    x = output[:,0]*opt.max_dist
    y = output[:,1]*opt.max_dist

    dt = opt.dt * opt.max_t
    vx = (x[1:] - x[:-1])/dt
    vy = (y[1:] - y[:-1])/dt

    ax = (vx[1:] - vx[:-1])/dt
    ay = (vy[1:] - vy[:-1])/dt
    vx = np.append(vx, vx[-1])
    vy = np.append(vy, vy[-1])
    ax = np.append(ax, ax[-1])
    ax = np.append(ax, ax[-1])
    ay = np.append(ay, ay[-1])
    ay = np.append(ay, ay[-1])
    
    a = np.sqrt(ax**2+ay**2)
    
    trajectory = {'time':plan_time, 'x':x, 'y':y, 'vx':vx, 'vy':vy, 'ax':ax, 'ay':ay, 'a':a}
    return trajectory

# town_list = ['Town01', 'Town02']
town_list = ['Town01']
weather_list = ['ClearNoon', 'ClearSunset']
def main():
    global global_vel, global_view_img, global_pcd, global_vehicle, global_plan_map, global_nav, state0, global_trajectory, global_collision, TOTAL_DIST
    client = carla.Client(config['host'], config['port'])
    client.set_timeout(config['timeout'])
    
    chosen_town = random.sample(town_list, 1)[0]
    chosen_weather = random.sample(weather_list, 1)[0]
    world = client.load_world(chosen_town)
    if chosen_town == 'Town02':
        TOTAL_DIST = 150
    if chosen_weather == 'ClearSunset':
        world.set_weather(carla.WeatherParameters.ClearSunset)
    else:
        world.set_weather(carla.WeatherParameters.ClearNoon)
    
    debug(info='Town: '+chosen_town+', weather: '+chosen_weather, info_type='message')
    if WRITE_LOG:
        os.makedirs('logs/'+chosen_town+'_'+chosen_weather+'/', exist_ok=True)
        log_file = open('logs/'+chosen_town+'_'+chosen_weather+'/'+str(time.time())+'.txt', 'a+')
        log_file.write('x(m)\ty(m)\tspeed(km/h)\tdist(m)\ttime(s)\n')

    blueprint = world.get_blueprint_library()
    world_map = world.get_map()
    
    vehicle = add_vehicle(world, blueprint, vehicle_type='vehicle.audi.a2')
    global_vehicle = vehicle

    spawn_points = world_map.get_spawn_points()
    waypoint_tuple_list = world_map.get_topology()
    origin_map = get_map(waypoint_tuple_list)

    agent = BasicAgent(vehicle, target_speed=MAX_SPEED)

    # prepare map
    destination = carla.Transform()
    destination.location = world.get_random_location_from_navigation()
    global_plan_map, destination = replan(agent, destination, copy.deepcopy(origin_map), spawn_points)

    vehicle.set_simulate_physics(True)
    physics_control = vehicle.get_physics_control()
    max_steer_angle = np.deg2rad(physics_control.wheels[0].max_steer_angle)

    sensor_dict = {
        'lidar':{
            'transform':carla.Transform(carla.Location(x=0.0, y=0.0, z=lidar_height)),
            'callback':lidar_callback,
            },
        'collision':{
            'transform':carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)),
            'callback':collision_callback,
        },
    }
    if DEBUG_VISUALIZATION:
        sensor_dict['camera:view'] = {
            # 'transform':carla.Transform(carla.Location(x=0.0, y=0.0, z=20.0), carla.Rotation(yaw=-90, pitch=-90)),
            'transform':carla.Transform(carla.Location(x=-10.0, y=0.0, z=15.0), carla.Rotation(pitch=-45)),
            'callback':view_image_callback,
            }

    sm = SensorManager(world, blueprint, vehicle, sensor_dict)
    sm.init_all()

    global_nav = get_nav(global_vehicle, global_plan_map)

    ctrller = CapacController(world, vehicle, MAX_SPEED)

    total_path_len = 0
    total_time = 0
    start_time = time.time()
    last_x = vehicle.get_location().x
    last_y = vehicle.get_location().y

    for total_steps in range(99999999999):
        if DEBUG_VISUALIZATION:
            while global_view_img is None or global_nav is None:
                time.sleep(0.001)
        else:
            while global_nav is None:
                time.sleep(0.001)

        if DEBUG_VISUALIZATION:
            visualize(global_view_img, global_nav)

        if close2dest(vehicle, destination) or total_path_len > TOTAL_DIST:
            print('get destination !!!')
            debug(info='Completion rate: 100%', info_type='success')
            if WRITE_LOG:
                log_file.write('Total dist:'+str(round(total_path_len,2))+'/'+str(round(total_path_len,2))+', completion rate: 100%\n')
                log_file.close()
                log_file = open('logs/'+chosen_town+'_'+chosen_weather+'/'+str(time.time())+'.txt', 'a+')
                log_file.write('x(m)\ty(m)\tspeed(km/h)\tdist(m)\ttime(m)\n')

            destination = carla.Transform()
            destination.location = world.get_random_location_from_navigation()
            global_plan_map, destination = replan(agent, destination, copy.deepcopy(origin_map), spawn_points)

            total_path_len = 0
            last_x = vehicle.get_location().x
            last_y = vehicle.get_location().y
            start_time = time.time()

        if global_collision:
            debug(info='Completion rate: '+str(round(100*total_path_len/TOTAL_DIST, 2))+'%', info_type='error')
            if WRITE_LOG:
                log_file.write('Total dist:'+str(round(total_path_len,2))+'/'+str(TOTAL_DIST)+', completion rate: '+str(round(100*total_path_len/TOTAL_DIST, 2))+'%\n')
                log_file.close()
                log_file = open('logs/'+chosen_town+'_'+chosen_weather+'/'+str(time.time())+'.txt', 'a+')
                log_file.write('x(m)\ty(m)\tspeed(km/h)\tdist(m)\ttime(m)\n')

            start_point = random.choice(spawn_points)
            vehicle.set_transform(start_point)
            global_plan_map, destination = replan(agent, destination, copy.deepcopy(origin_map), spawn_points)

            start_waypoint = agent._map.get_waypoint(agent._vehicle.get_location())
            end_waypoint = agent._map.get_waypoint(destination.location)

            route_trace = agent._trace_route(start_waypoint, end_waypoint)
            start_point.rotation = route_trace[0][0].transform.rotation
            vehicle.set_transform(start_point)
            time.sleep(0.1)
            global_collision = False

            total_path_len = 0
            last_x = vehicle.get_location().x
            last_y = vehicle.get_location().y
            start_time = time.time()

        vel = vehicle.get_velocity()
        global_vel = np.sqrt(vel.x**2+vel.y**2+vel.z**2)

        voxel_index = pcd2voxel(global_pcd)

        nav = get_nav(global_vehicle, global_plan_map)
        global_nav = nav        

        nav_tf = nav_transforms(Image.fromarray(nav).convert('L'))

        input_data = {
            'lidar': voxel_index,
            'nav': nav_tf.numpy(),
        }

        global_trajectory = get_trajectory(input_data)

        control_time = time.time()
        dt = control_time - global_trajectory['time']
        index = int((dt/opt.max_t)//opt.dt) + 1#3

        if index > 0.99/opt.dt:
            continue
    
        control = ctrller.run_step(global_trajectory, index, state0)
        vehicle.apply_control(control)

        # control = agent.run_step()
        # control.manual_gear_shift = False
        # vehicle.apply_control(control)

        x = vehicle.get_location().x
        y = vehicle.get_location().y
        dist = np.sqrt((x-last_x)*(x-last_x) + (y-last_y)*(y-last_y))
        total_path_len += dist
        last_x = x
        last_y = y
        # print('total running dist:', round(total_path_len, 1), 'm,  speed:', round(3.6*np.sqrt(vel.x**2+vel.y**2), 1), 'km/h')
        vel = vehicle.get_velocity()
        speed = 3.6*np.sqrt(vel.x**2+vel.y**2)
        total_time = time.time() - start_time
        debug(info='total running dist: '+str(round(total_path_len, 1))+' m,  speed: '+str(round(speed, 1))+' km/h  used time: '+str(round(total_time, 2))+' s', info_type='message')
        if WRITE_LOG:
            log_file.write(str(round(x,2))+'\t'+str(round(y,2))+'\t'+str(round(speed,2))+'\t'+str(round(total_path_len,2))+'\t'+str(round(total_time, 2))+'\n')

        if DEBUG_VISUALIZATION:
            for i in range(0, len(global_trajectory['x']), 1):
                xi = global_trajectory['x'][i]
                yi = global_trajectory['y'][i]
                
                pose = vehicle.get_transform()
                location = pose.location
                yaw = np.deg2rad(pose.rotation.yaw)
                _x = xi*np.cos(yaw) - yi*np.sin(yaw)
                _y = xi*np.sin(yaw) + yi*np.cos(yaw)


                localtion = carla.Location(x = location.x+_x, y=location.y+_y, z=0.2)
                world.debug.draw_point(localtion, size=0.2, color=carla.Color(255,0,0), life_time=0.1)

        # time.sleep(1/30.)
    if DEBUG_VISUALIZATION:
        cv2.destroyAllWindows()

    sm.close_all()
    vehicle.destroy()
        
if __name__ == '__main__':
    main()