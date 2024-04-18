
import numpy as np
from informer import Informer
from copy import deepcopy as cp

pixs_per_meter_heignt = 10
length, width = 512, 512
height = int(2*pixs_per_meter_heignt)

global_voxel = np.zeros((height, length, width), np.float32)

global_lidar = None
global_nav = None

def parse_lidar(message, robot_id):
    global global_lidar, global_voxel
    print('get msg in global_lidar', len(message))
    puv = np.frombuffer(message, dtype=np.int64).reshape(3, -1)
    # print('puv shape', puv.shape)
    p = puv[0]
    u = puv[1]
    v = puv[2]
    voxel = cp(global_voxel)
    voxel.fill(0.)
    voxel[p,u,v] = 1.
    global_lidar = voxel
    # print('global_lidar shape', global_lidar.shape)


def parse_nav(message, robot_id):
    global global_nav
    print('get msg in global_nav', len(message))
    global_nav = message
    global_nav = np.frombuffer(message, dtype=np.float32).reshape(1, 200, 400)
    # print('nav shape', global_nav.shape)

class Server(Informer):
    def send_traj(self, message):
        self.send(message, 'traj')
    
    def lidar_recv(self):
        self.recv('lidar', parse_lidar)

    def nav_recv(self):
        self.recv('nav', parse_nav)

ifm = Server(config = 'config_atlas.yaml')

while True:
    if global_lidar is not None and global_nav is not None:
        msg = np.random.randn(20, 2).astype(np.float32)
        msg = msg.tobytes()
        ifm.send_traj(msg)
        print('send traj !!!')

        global_lidar = None
        global_nav = None