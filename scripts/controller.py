
import time
import numpy as np
import matplotlib.pyplot as plt

import carla

class Param(object):
    def __init__(self):
        '''vehicle'''
        self.L = 2.405
        self.max_acceleration = 1.0
        self.min_acceleration = -1.0
        self.max_steer = 1.0

        '''PID'''
        self.v_pid_Kp = 1.00
        self.v_pid_Ki = 0.00
        self.v_pid_Kd = 0.00

        '''RWPF'''
        self.k_k = 1.235
        self.k_theta = 0.456
        self.k_e = 0.11

        '''debug'''
        self.curvature_factor = 1.0

def plotArrow2D(x, y, theta, length=1.0, width=0.5, fc='r', ec='k'):  # pragma: no cover
    plt.arrow(x, y, length * np.cos(theta), length * np.sin(theta),
              fc=fc, ec=ec, head_width=width, head_length=width)

def vector3DToArray(vec):
    return np.array([vec.x, vec.y, vec.z]).reshape(3,1)

def pi2pi(theta, theta0=0.0):
    return (theta + np.pi) % (2 * np.pi) - np.pi

class State(object):
    def __init__(self, frame_id, time_stamp, **kwargs):
        self.frame_id = frame_id
        self.time_stamp = time_stamp

        self.x = float(kwargs.get('x', 0))
        self.y = float(kwargs.get('y', 0))
        self.z = float(kwargs.get('z', 0))

        self.theta = pi2pi(float(kwargs.get('theta', 0)))

        self.k = float(kwargs.get('k', 0))

        self.s = float(kwargs.get('s', 0))
        self.v = float(kwargs.get('v', 0))
        self.a = float(kwargs.get('a', 0))
        self.t = float(kwargs.get('t', 0))

        self.velocity = kwargs.get('velocity', np.zeros((3,1))).astype(np.float64)
        self.acceleration = kwargs.get('acceleration', np.zeros((3,1))).astype(np.float64)
        

    def __str__(self):
        obj = 'frame_id: {}, time_stamp: {}, x: {}, y: {}, theta: {}, v: {}'.format(self.frame_id, self.time_stamp, self.x, self.y, self.theta, self.v)
        return obj


    def distance_xyz(self, state):
        dx = self.x - state.x
        dy = self.y - state.y
        dz = self.z - state.z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def distance_xy(self, state):
        dx = self.x - state.x
        dy = self.y - state.y
        return np.sqrt(dx**2 + dy**2)

    def distance_xytheta(self, state):
        dx = self.x - state.x
        dy = self.y - state.y
        dtheta = self.theta - state.theta
        return np.sqrt(dx**2 + dy**2 + dtheta**2)

    def delta_theta(self, state):
        dx = self.x - state.x
        dy = self.y - state.y
        return np.arctan2(dy, dx)


    def find_nearest_state_in(self, states):
        min_dist = float('inf')
        min_state = None
        for state in states:
            d = self.distance(state)
            if d < min_dist:
                min_dist = d
                min_state = state
        return min_state

    def find_nearest_waypoint_in(self, waypoints):
        mind = float('inf')
        nearest_wp = None
        for waypoint in waypoints:
            d = self.distance(waypoint)
            if d < mind:
                mind = d
                nearest_wp = waypoint
        return nearest_wp, mind


    def world_to_local_2D(self, state0, local_frame_id):
        if self.frame_id != state0.frame_id:
            print('carla_utils/augment/State: Wrong1')

        x_world, y_world, theta_world = self.x, self.y, self.theta
        x0, y0, theta0 = state0.x, state0.y, state0.theta
        delta_theta = pi2pi(self.theta - state0.theta)

        x_local = (x_world-x0)*np.cos(theta0) + (y_world-y0)*np.sin(theta0)
        y_local =-(x_world-x0)*np.sin(theta0) + (y_world-y0)*np.cos(theta0)

        local_state = State(
                local_frame_id, self.time_stamp, x=x_local, y=y_local, theta=delta_theta,
                z=self.z, k=self.k, s=self.s, v=self.v, t=self.t
                )
        return local_state

    def local_to_world_2D(self, state0, world_frame_id):
        if state0.frame_id != world_frame_id:
            print('carla_utils/augment/State: Wrong2')

        x_local, y_local, theta_local = self.x, self.y, self.theta
        x0, y0, theta0 = state0.x, state0.y, state0.theta

        x_world = x0 + x_local*np.cos(theta0) - y_local*np.sin(theta0)
        y_world = y0 + x_local*np.sin(theta0) + y_local*np.cos(theta0)
        theta_world = pi2pi(theta_local + theta0)

        world_state = State(
                world_frame_id, self.time_stamp, x=x_world, y=y_world, theta=theta_world,
                z=self.z, k=self.k, s=self.s, v=self.v, a=self.a, t=self.t
                )
        return world_state

    def visualInMatplotlib(self, style='arrow', fmt='og', color='red'):
        if style == 'point':
            plt.plot(self.x, self.y, fmt, color=color)
        elif style == 'arrow':
            plotArrow2D(self, fc=color, ec=color)


def getActorState(frame_id, time_stamp, actor):
    location = actor.get_location()
    x, y, z = location.x, location.y, location.z
    theta = np.deg2rad(actor.get_transform().rotation.yaw)
    velocity = vector3DToArray(actor.get_velocity())
    v = np.linalg.norm(velocity)
    acceleration = vector3DToArray(actor.get_acceleration())
    a = np.linalg.norm(acceleration)
    return State(frame_id, time_stamp, x=x, y=y, z=z, theta=theta, v=v, a=a)


class CapacController(object):
    def __init__(self, world, vehicle, frequency):
        '''parameter'''
        config = Param()
        self.world = world
        self.vehicle = vehicle
        self.L = config.L
        self.max_steer = config.max_steer
        self.dt = 1. / frequency

        '''debug'''
        self.curvature_factor = config.curvature_factor

        '''PID'''
        self.Kp, self.Ki, self.Kd = config.v_pid_Kp, config.v_pid_Ki, config.v_pid_Kd
        self.max_a = config.max_acceleration
        self.min_a = config.min_acceleration
        self.last_error = 0
        self.sum_error = 0

        '''RWPF'''
        self.k_k, self.k_theta, self.k_e = config.k_k, config.k_theta, config.k_e
    

    def run_step(self, trajectory, index, state0):
        '''
        Args:
            trajectory = {'time':plan_time, 'x':x, 'y':y, 'vx':vx, 'vy':vy, 'ax':ax, 'ay':ay, 'a':a}
        '''
        time_stamp = time.time()
        current_state = getActorState('odom', time_stamp, self.vehicle)
        current_state = current_state.world_to_local_2D(state0, 'base_link')

        x, y, vx, vy = trajectory['x'][index], trajectory['y'][index], trajectory['vx'][index],  trajectory['vy'][index]
        ax, ay = trajectory['ax'][index], trajectory['ay'][index]
        a, t = trajectory['a'][index], trajectory['time']
        theta, v = np.arctan2(vy, vx), np.hypot(vx, vy)
        k = (vx*ay-vy*ax)/(v**3)
        target_state = State('base_link',time_stamp, x=x, y=y+0.08, theta=theta, v=v, a=a, k=k, t=t)
        target_state.y = target_state.y
        
        steer = self.rwpf(current_state, target_state)
        throttle, brake = self.pid(current_state, target_state)
        target_state.k = k * self.curvature_factor
        
        # debug
        global_target = target_state.local_to_world_2D(state0, 'odom')
        localtion = carla.Location(x = global_target.x, y=global_target.y, z=2.0)
        self.world.debug.draw_point(localtion, size=0.2, color=carla.Color(0,255,0), life_time=0.1)
        # throttle, brake = 1, 0
        throttle += 0.3
        throttle = np.clip(throttle, 0., 1.)
    
        # if throttle > 0 and abs(current_state.v) < 1.0 and abs(target_state.v) < 1.0:
        if throttle > 0 and abs(current_state.v) < 1.5 and abs(target_state.v) < 1.5:
            throttle = 0.
            brake = 1.
            steer = 0.
            
        steer = steer*1.5 #2.5
        return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)

    def pid(self, current_state, target_state):
        v_current = current_state.v
        v_target = target_state.v
        error = v_target - v_current
        
        acceleration = self.Kp * error
        acceleration += self.Ki * self.sum_error * self.dt
        acceleration += self.Kd * (error - self.last_error) / self.dt

        self.last_error = error
        self.sum_error += error
        '''eliminate drift'''
        if abs(self.sum_error) > 10:
            self.sum_error = 0.0

        throttle = np.clip(acceleration, 0, self.max_a)
        brake = -np.clip(acceleration, self.min_a, 0)*0.3
        
        return throttle, brake


    def rwpf(self, current_state, target_state):
        xr = target_state.x
        yr = target_state.y
        thetar = target_state.theta
        vr = target_state.v
        kr = target_state.k

        dx = current_state.x - xr
        dy = current_state.y - yr
        tx = np.cos(thetar)
        ty = np.sin(thetar)
        e = dx*ty - dy*tx
        theta_e = pi2pi(current_state.theta - thetar)

        alpha = 1.8
        
        w1 = self.k_k * vr*kr*np.cos(theta_e)
        w2 = - self.k_theta * np.fabs(vr)*theta_e
        w3 = (self.k_e*vr*np.exp(-theta_e**2/alpha))*e
        w = (w1+w2+w3)*0.8
        if current_state.v < 0.02:
            steer = 0
        else:
            steer = np.arctan2(w*self.L, current_state.v) * 2 / np.pi * self.max_steer
        
        #print(w, steer)
        return steer