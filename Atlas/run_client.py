"""
-*- coding:utf-8 -*-
"""
import acl
import os, sys
import numpy as np
import time
sys.path.append('../thirdparty/acllite')
from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel

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

class TrajectoryModel(object):
    def __init__(self, lidar_encoder_path, nav_encoder_path, decoder_path):
        self.device_id = 0
        self.lidar_encoder_path = lidar_encoder_path
        self.nav_encoder_path = nav_encoder_path
        self.decoder_path = decoder_path

        self.lidar_encoder = None
        self.nav_encoder = None
        self.decoder = None
        self.t = np.expand_dims(np.arange(0, 0.99, 0.05, dtype=np.float32), axis=1)
        self.init_resource()

    def __del__(self):
        print("Model release source success")

    def init_resource(self):
        self.acl_resource = AclLiteResource()
        self.acl_resource.init()

        self.lidar_encoder = AclLiteModel(self.lidar_encoder_path)
        self.nav_encoder = AclLiteModel(self.nav_encoder_path)
        self.decoder = AclLiteModel(self.decoder_path)

    def inference(self, lidar, nav):
        lidar_inference_results = self.lidar_encoder.execute([lidar, ])
        lidar_feature = lidar_inference_results[0]

        nav_inference_results = self.nav_encoder.execute([nav, ])
        nav_feature = nav_inference_results[0]

        single_latent = np.hstack([lidar_feature, nav_feature])
        single_latent = np.expand_dims(single_latent, axis=0)
        latent = np.repeat(single_latent, self.t.shape[0], axis=1)
        latent = latent.reshape(single_latent.shape[0] * self.t.shape[0], single_latent.shape[-1])
        decoder_input = np.hstack([latent, self.t])

        decoder_inference_results = self.decoder.execute([decoder_input, ])
        output = decoder_inference_results[0]

        return output
    
if __name__ == '__main__':
    ifm = Server(config = 'config_atlas.yaml')
    models = TrajectoryModel('lidar_encoder.om', 'nav_encoder.om', 'decoder.om')
    while True:
        if global_lidar is not None and global_nav is not None:
            t1 = time.time()
            output = models.inference(global_lidar, global_nav)
            print('generate trajectory:', output.shape)
            msg = output.tobytes()
            ifm.send_traj(msg)
            t2 = time.time()
            print('inference time:', round(1000*(t2-t1), 2), 'ms')

            global_lidar = None
            global_nav = None
        else:
            time.sleep(0.001)
