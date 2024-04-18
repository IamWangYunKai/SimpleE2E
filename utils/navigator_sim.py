#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import random
import numpy as np
import PIL.Image as Image
from PIL import ImageDraw

scale = 12.0
x_offset = 800#1500#2500
y_offset = 1000#0#3000

def get_random_destination(spawn_points):
    return random.sample(spawn_points, 1)[0]
    
def get_map(waypoint_tuple_list):
    origin_map = np.zeros((6000, 6000, 3), dtype="uint8")
    origin_map.fill(255)
    origin_map = Image.fromarray(origin_map)
    return origin_map

def draw_route(agent, destination, origin_map, spawn_points):
    start_waypoint = agent._map.get_waypoint(agent._vehicle.get_location())
    end_waypoint = agent._map.get_waypoint(destination.location)
    while True:
        new_destination = destination
        route_trace = agent._trace_route(start_waypoint, end_waypoint)

        route_trace_list = []
        dist = 0.

        road_ids = [route_trace[0][0].road_id]
        continue_flag = False
        for i in range(len(route_trace)-1):
            road_id = route_trace[i][0].road_id
            if road_id != road_ids[-1]:
                if road_id in road_ids:
                    continue_flag = True
                    destination = random.choice(spawn_points)
                    end_waypoint = agent._map.get_waypoint(destination.location)
                    break
                else:
                    road_ids.append(road_id)
        if continue_flag: continue

        for i in range(len(route_trace)-1):
            dist += np.sqrt((route_trace[i][0].transform.location.x-route_trace[i+1][0].transform.location.x)**2+(route_trace[i][0].transform.location.y-route_trace[i+1][0].transform.location.y)**2)
            x = scale*route_trace[i][0].transform.location.x+x_offset
            y = scale*route_trace[i][0].transform.location.y+y_offset
            route_trace_list.append(x)
            route_trace_list.append(y)

        if (dist < 50):
            destination = random.choice(spawn_points)
            end_waypoint = agent._map.get_waypoint(destination.location)
        else:
            break

    draw = ImageDraw.Draw(origin_map)
    draw.line(route_trace_list, 'red', width=30)
    return origin_map, new_destination

def get_nav(vehicle, plan_map, town=1):
    if town == 1:
        x_offset = 800
        y_offset = 1000
    elif town == 2:
        x_offset = 1500
        y_offset = 0
    x = int(scale*vehicle.get_location().x + x_offset)
    y = int(scale*vehicle.get_location().y + y_offset)
    _nav = plan_map.crop((x-400,y-400, x+400, y+400))
    
    im_rotate = _nav.rotate(vehicle.get_transform().rotation.yaw+90)

    im_rotate = im_rotate.transpose(Image.FLIP_LEFT_RIGHT)
    WIDTH = 160
    HIGHT = 80
    nav = im_rotate.crop((_nav.size[0]//2-WIDTH, _nav.size[1]//2-2*HIGHT, _nav.size[0]//2+WIDTH, _nav.size[1]//2))
    nav = cv2.cvtColor(np.asarray(nav), cv2.COLOR_BGR2RGB)

    return nav

def get_big_nav(vehicle, plan_map):
    x = int(scale*vehicle.get_location().x + x_offset)
    y = int(scale*vehicle.get_location().y + y_offset)
    _nav = plan_map.crop((x-400,y-400, x+400, y+400))
    
    r = 20
    draw = ImageDraw.Draw(_nav)
    draw.ellipse((_nav.size[0]//2-r, _nav.size[1]//2-r, _nav.size[0]//2+r, _nav.size[1]//2+r), fill='green', outline='green', width=10)
    
    im_rotate = _nav.rotate(vehicle.get_transform().rotation.yaw+90)
    #nav = im_rotate
    nav = im_rotate.crop((0, 0, _nav.size[0], _nav.size[1]//2))
    nav = cv2.cvtColor(np.asarray(nav), cv2.COLOR_BGR2RGB)
    return nav

def replan(agent, destination, origin_map, spawn_points):
    agent.set_destination((destination.location.x,
                           destination.location.y,
                           destination.location.z))
    plan_map, new_destination = draw_route(agent, destination, origin_map, spawn_points)
    agent.set_destination((new_destination.location.x,
                           new_destination.location.y,
                           new_destination.location.z))
    return plan_map, new_destination

def replan2(agent, destination, origin_map):
    agent.set_destination(agent.vehicle.get_location(), destination.location, clean=True)
    plan_map = draw_route(agent, destination, origin_map)
    return plan_map
    
def close2dest(vehicle, destination, dist=30):
    return destination.location.distance(vehicle.get_location()) < dist