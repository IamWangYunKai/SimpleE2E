#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '../'))

import torch
from learning.model import LidarEncoder, NavEncoder, Generator

device = torch.device('cpu')

pixs_per_meter = 5
pixs_per_meter_heignt = 10
length, width = 512, 512
height = int(2*pixs_per_meter_heignt)

IMG_HEIGHT = 200
IMG_WIDTH = 400

lidar_encoder = LidarEncoder(in_channels=height).to(device)
nav_encoder = NavEncoder(input_dim=1, out_dim=256).to(device)
decoder = Generator(input_dim=512+256+1, output_dim=2).to(device)

lidar_encoder.load_state_dict(torch.load('../ckpt/lidar_encoder.pth'))
nav_encoder.load_state_dict(torch.load('../ckpt/nav_encoder.pth'))
decoder.load_state_dict(torch.load('../ckpt/decoder.pth'))


lidar_encoder_export_onnx_file = 'lidar_encoder.onnx'
nav_encoder_export_onnx_file = 'nav_encoder.onnx'
decoder_export_onnx_file = 'decoder.onnx'


dummy_lidar_input = torch.randn(1, 20, 512, 512)
dummy_nav_input = torch.randn(1, 1, 200, 400)
dummy_decoder_input = torch.randn(20, 769)

torch.onnx.export(
    lidar_encoder,
    dummy_lidar_input,
    lidar_encoder_export_onnx_file,
    opset_version=10,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input":{0:"batch_size"},
                    "output":{0:"batch_size"}})

torch.onnx.export(
    nav_encoder,
    dummy_nav_input,
    nav_encoder_export_onnx_file,
    opset_version=10,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input":{0:"batch_size"},
                    "output":{0:"batch_size"}})


torch.onnx.export(
    decoder,
    dummy_decoder_input,
    decoder_export_onnx_file,
    opset_version=10,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input":{0:"batch_size"},
                    "output":{0:"batch_size"}})