#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import random

config = {
    'host': 'localhost',
    'port': 2000,
    'timeout': 5.0,
    'camera':{
        'role_name': 'front',
        'img_length': 1280,#400,
        'img_width': 640,#200,
        'fov': 120,
        'fps': 30,
        },
    'lidar':{
        'channels': 32,
        'rpm': 30,
        'sensor_tick':0.05,
        'pps': 100000,
        # 'range': 5000,# <= 0.9.5
        'range': 50,
        'lower_fov': -30,
        'upper_fov': 15,
        },
    'imu':{
        'fps': 400,
        },
    'gnss':{
        'fps': 20,
        },
}

def load(path='~/CARLA_0.9.9.4'):
    try:
        sys.path.append(path+'/PythonAPI')
        sys.path.append(glob.glob(path+'/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except:
        print('Fail to load carla library')
        
def set_weather(world, weather):
    world.set_weather(weather)
    return weather

def add_vehicle(world, blueprint, vehicle_type='vehicle.bmw.grandtourer'):
    bp = random.choice(blueprint.filter(vehicle_type))
    if bp.has_attribute('color'):
        color = random.choice(bp.get_attribute('color').recommended_values)
        bp.set_attribute('color', color)
    transform = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, transform)
    return vehicle