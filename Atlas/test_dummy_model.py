"""
-*- coding:utf-8 -*-
"""
import acl
import os, sys
import numpy as np
import time
sys.path.append('./acllite')
from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel

class MetaModel(object):
    def __init__(self, model_path):
        self.device_id = 0
        self.model_path = model_path
        self.init_resource()

    def __del__(self):
        print("Model release source success")

    def init_resource(self):
        self.acl_resource = AclLiteResource()
        self.acl_resource.init()
        self.model_process = AclLiteModel(self.model_path)

    def inference(self, input_data):
        inference_results = self.model_process.execute([input_data, ])
        output = inference_results[0]
        return output

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
    models = TrajectoryModel('lidar_encoder.om', 'nav_encoder.om', 'decoder.om')
    for i in range(10):
        # dummy_lidar_input = np.random.randn(1, 20, 512, 512).astype(np.float32)
        # dummy_nav_input = np.random.randn(1, 1, 200, 400).astype(np.float32)
        dummy_lidar_input = i*np.ones((1, 20, 512, 512), dtype=np.float32)
        dummy_nav_input = i*np.ones((1, 1, 200, 400), dtype=np.float32)
        t1 = time.time()
        output = models.inference(dummy_lidar_input, dummy_nav_input)
        t2 = time.time()
        print('time:', round(1000*(t2-t1), 2))
        print('output:', output.shape)
