#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '../'))

import simulator
simulator.load('~/CARLA_0.9.9.4')
import carla
sys.path.append('~/CARLA_0.9.9.4//PythonAPI/carla')
from agents.navigation.basic_agent import BasicAgent

from simulator import config, set_weather, add_vehicle
from simulator.sensor_manager import SensorManager
from utils.navigator_sim import get_map, get_nav, replan, close2dest
from utils import add_alpha_channel, debug
import matplotlib.pyplot as plt
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

DEBUG_VISUALIZATION = True
TOTAL_DIST = 300
WRITE_LOG = False

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


lidar_height = 2.5

pixs_per_meter = 5
pixs_per_meter_heignt = 10
length, width = 512, 512
height = int(2*pixs_per_meter_heignt)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import grad

from learning.model import LidarEncoder, NavEncoder, Generator

# device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

lidar_encoder = LidarEncoder(in_channels=height).to(device)
nav_encoder = NavEncoder(input_dim=1, out_dim=256).to(device)
decoder = Generator(input_dim=512+256+1, output_dim=2).to(device)

lidar_encoder.load_state_dict(torch.load('../ckpt/lidar_encoder.pth'))
nav_encoder.load_state_dict(torch.load('../ckpt/nav_encoder.pth'))
decoder.load_state_dict(torch.load('../ckpt/decoder.pth'))


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

def get_trajectory(batch):
    global global_trajectory, global_plan_time, global_vehicle, state0
    global_plan_time = time.time()
    plan_time = global_plan_time
    transform = global_vehicle.get_transform()
    state0 = getActorState('odom', global_plan_time, global_vehicle)
    state0.x = transform.location.x
    state0.y = transform.location.y
    state0.z = transform.location.z
    state0.theta = np.deg2rad(transform.rotation.yaw)
    
    batch['lidar'] = batch['lidar']
    batch['nav'] = batch['nav']
    batch['t'] = batch['t'].view(-1,1)
    batch['t'].requires_grad = True

    lidar_feature = lidar_encoder(batch['lidar'])
    nav_feature = nav_encoder(batch['nav'])
    single_latent = torch.cat([lidar_feature, nav_feature], dim=1)
    single_latent = single_latent.unsqueeze(1)
    latent = single_latent.expand(single_latent.shape[0], batch['t'].shape[0], single_latent.shape[-1])
    latent = latent.reshape(single_latent.shape[0] * batch['t'].shape[0], single_latent.shape[-1])
    decoder_input = torch.cat([latent, batch['t']], dim=1)
    output = decoder(decoder_input)

    vx = grad(output[:,0].sum(), batch['t'], create_graph=True)[0][:,0]*(opt.max_dist/opt.max_t)
    vy = grad(output[:,1].sum(), batch['t'], create_graph=True)[0][:,0]*(opt.max_dist/opt.max_t)
    
    ax = grad(vx.sum(), batch['t'], create_graph=True)[0][:,0]/opt.max_t
    ay = grad(vy.sum(), batch['t'], create_graph=True)[0][:,0]/opt.max_t

    output_axy = torch.cat([ax.unsqueeze(1), ay.unsqueeze(1)], dim=1)

    x = output[:,0]*opt.max_dist
    y = output[:,1]*opt.max_dist

    theta_a = torch.atan2(ay, ax)
    theta_v = torch.atan2(vy, vx)
    sign = torch.sign(torch.cos(theta_a-theta_v))
    a = torch.mul(torch.norm(output_axy, dim=1), sign.flatten()).unsqueeze(1)

    
    vx = vx.data.cpu().numpy()
    vy = vy.data.cpu().numpy()
    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()
    ax = ax.data.cpu().numpy()
    ay = ay.data.cpu().numpy()
    a = a.data.cpu().numpy()

    trajectory = {'time':plan_time, 'x':x, 'y':y, 'vx':vx, 'vy':vy, 'ax':ax, 'ay':ay, 'a':a}

    return trajectory

town_list = ['Town01', 'Town01', 'Town02']
weather_list = ['ClearNoon', 'ClearNoon', 'ClearSunset']
def main():
    global global_vel, global_view_img, global_pcd, global_vehicle, global_plan_map, global_nav, state0, global_trajectory, global_collision
    client = carla.Client(config['host'], config['port'])
    client.set_timeout(config['timeout'])
    
    # world = client.load_world('Town01')
    # world = client.load_world('Town02')
    # world.set_weather(carla.WeatherParameters.ClearNoon)
    # world.set_weather(carla.WeatherParameters.ClearSunset)
    chosen_town = random.sample(town_list, 1)[0]
    chosen_weather = random.sample(weather_list, 1)[0]
    world = client.load_world(chosen_town)
    if chosen_weather == 'ClearSunset':
        world.set_weather(carla.WeatherParameters.ClearSunset)
    else:
        world.set_weather(carla.WeatherParameters.ClearNoon)
    
    debug(info='Town: '+chosen_town+', weather: '+chosen_weather, info_type='message')
    if WRITE_LOG:
        os.makedirs('logs/'+chosen_town+'_'+chosen_weather+'/', exist_ok=True)
        log_file = open('logs/'+chosen_town+'_'+chosen_weather+'/'+str(time.time())+'.txt', 'a+')
        log_file.write('x(m)\ty(m)\tspeed(km/h)\tdist(m)\ttime(m)\n')

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
            'transform':carla.Transform(carla.Location(x=0.0, y=0.0, z=20.0), carla.Rotation(yaw=-90, pitch=-90)),
            # 'transform':carla.Transform(carla.Location(x=-3.0, y=0.0, z=6.0), carla.Rotation(pitch=-45)),
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

            destination = carla.Transform()
            destination.location = world.get_random_location_from_navigation()
            global_plan_map, destination = replan(agent, destination, copy.deepcopy(origin_map), spawn_points)

            total_path_len = 0
            last_x = vehicle.get_location().x
            last_y = vehicle.get_location().y

        if global_collision:
            debug(info='Completion rate: '+str(round(100*total_path_len/TOTAL_DIST, 2))+'%', info_type='error')
            if WRITE_LOG:
                log_file.write('Total dist:'+str(round(total_path_len,2))+'/'+str(TOTAL_DIST)+', completion rate: '+str(round(100*total_path_len/TOTAL_DIST, 2))+'%\n')
                log_file.close()
                log_file = open('logs/'+chosen_town+'_'+chosen_weather+'/'+str(time.time())+'.txt', 'a+')

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

        # control = agent.run_step()
        # control.manual_gear_shift = False
        # vehicle.apply_control(control)

        vel = vehicle.get_velocity()
        global_vel = np.sqrt(vel.x**2+vel.y**2+vel.z**2)

        voxel = pcd2voxel(global_pcd)
        lidar_data = torch.from_numpy(voxel)
        nav = get_nav(global_vehicle, global_plan_map)
        global_nav = nav

        t = torch.arange(0, 0.99, opt.dt).unsqueeze(1)
        points_num = len(t)
        
        v0_array = torch.FloatTensor(np.array([vel.x/opt.max_speed, vel.y/opt.max_speed]*points_num).astype(np.float32)).T
        if np.sqrt(vel.x**2+vel.y**2) < 3:
            vel.x = 3
            v0_array = torch.FloatTensor(np.array([vel.x/opt.max_speed, vel.y/opt.max_speed]*points_num).astype(np.float32)).T

        nav_tf = nav_transforms(Image.fromarray(nav).convert('L'))

        batch = {
            'lidar': lidar_data.unsqueeze(0).to(device),
            'nav': nav_tf.unsqueeze(0).to(device),
            'v0_array': v0_array.unsqueeze(0).to(device),
            't': t.to(device),
        }

        global_trajectory = get_trajectory(batch)

        control_time = time.time()
        dt = control_time - global_trajectory['time']
        index = int((dt/opt.max_t)//opt.dt) + 3

        if index > 0.99/opt.dt:
            continue
    
        control = ctrller.run_step(global_trajectory, index, state0)
        vehicle.apply_control(control)


        x = vehicle.get_location().x
        y = vehicle.get_location().y
        dist = np.sqrt((x-last_x)*(x-last_x) + (y-last_y)*(y-last_y))
        total_path_len += dist
        last_x = x
        last_y = y

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
                world.debug.draw_point(localtion, size=0.2, color=carla.Color(255,0,0), life_time=0.01)

    if DEBUG_VISUALIZATION:
        cv2.destroyAllWindows()

    sm.close_all()
    vehicle.destroy()
        
if __name__ == '__main__':
    main()