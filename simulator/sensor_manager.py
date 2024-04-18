#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from simulator import config
from utils import debug, Singleton

# https://carla.readthedocs.io/en/latest/ref_sensors/
def add_camera(world, blueprint, vehicle, transform):
    camera_bp = blueprint.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(config['camera']['img_length']))
    camera_bp.set_attribute('image_size_y', str(config['camera']['img_width']))
    camera_bp.set_attribute('fov', str(config['camera']['fov']))
    camera_bp.set_attribute('sensor_tick', str(1./config['camera']['fps']))
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    return camera

def add_semantic(world, blueprint, vehicle, transform):
    semantic_bp = blueprint.find('sensor.camera.semantic_segmentation')
    semantic_bp.set_attribute('image_size_x', str(config['camera']['img_length']))
    semantic_bp.set_attribute('image_size_y', str(config['camera']['img_width']))
    semantic_bp.set_attribute('fov', str(config['camera']['fov']))
    semantic_bp.set_attribute('sensor_tick', str(1./config['camera']['fps']))
    semantic = world.spawn_actor(semantic_bp, transform, attach_to=vehicle)
    return semantic

def add_lidar(world, blueprint, vehicle, transform):
    lidar_bp = blueprint.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', str(config['lidar']['channels']))
    lidar_bp.set_attribute('rotation_frequency', str(config['lidar']['rpm']))
    lidar_bp.set_attribute('points_per_second', str(config['lidar']['pps']))
    lidar_bp.set_attribute('sensor_tick', str(config['lidar']['sensor_tick']))
    lidar_bp.set_attribute('range', str(config['lidar']['range']))
    lidar_bp.set_attribute('lower_fov', str(config['lidar']['lower_fov']))
    lidar_bp.set_attribute('upper_fov', str(config['lidar']['upper_fov']))
    lidar = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
    return lidar
    
def add_imu(world, blueprint, vehicle, transform):
    imu_bp = blueprint.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', str(1./config['imu']['fps']))
    imu = world.spawn_actor(imu_bp, transform, attach_to=vehicle)
    return imu

def add_gnss(world, blueprint, vehicle, transform):
    gnss_bp = blueprint.find('sensor.other.gnss')
    gnss_bp.set_attribute('sensor_tick', str(1./config['gnss']['fps']))
    gnss = world.spawn_actor(gnss_bp, transform, attach_to=vehicle)
    return gnss

def add_collision(world, blueprint, vehicle, transform):
    collision_bp = blueprint.find('sensor.other.collision')
    collision = world.spawn_actor(collision_bp, transform, attach_to=vehicle)
    return collision

class SensorManager(Singleton):
    def __init__(self, world, blueprint, vehicle, param_dict):
        self.world = world
        self.blueprint = blueprint
        self.vehicle = vehicle
        self.param_dict = param_dict
        self.sensor_dict = {}

        self.known_sensors = ['camera', 'lidar', 'imu', 'gnss', 'semantic', 'collision']

    def init(self, key):
        if key in self.param_dict:
            sensor_type = self.get_type(key)
            sensor = globals()['add_' + sensor_type](
                    self.world,
                    self.blueprint,
                    self.vehicle,
                    self.param_dict[key]['transform'])
            sensor.listen(lambda data: self.param_dict[key]['callback'](data))
            self.sensor_dict[key] = sensor
            debug(info=key+' successfully initialized !', info_type='success')
        else:
            debug(info='Unknown sensor '+ str(key), info_type='error')
            return None	

    def init_all(self):
        for key in self.param_dict:
            try:
                self.init(key)
            except:
                debug(info=str(key)+' initialize failed', info_type='error')

    def close_all(self):
        for key in self.param_dict:
            try:
                self.sensor_dict[key].destroy()
                debug(info=str(key)+' closed', info_type='success')
            except:
                debug(info=str(key)+' has no attribute called \'close\'', info_type='message')
        
    def __del__(self):
        self.close_all()
    
	# get sensor instance
    def __getitem__(self, key):
        if key in self.sensor_dict:
            return self.sensor_dict[key]
        else:
            debug(info='No sensor called '+ str(key), info_type='error')
            return None
    
    # set sensor param
    def __setitem__(self, key, value):
        if key in self.param_dict:
            self.param_dict[key] = value
            return True
        else:
            debug(info='No sensor called '+ str(key), info_type='error')
            return None

    def get_type(self, key):
        sensor_type = key.split(':')[0]
        if sensor_type in self.known_sensors:
            return sensor_type
        else:
            debug(info='Unknown sensor type '+ str(key), info_type='error')
            return None