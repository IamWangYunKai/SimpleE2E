import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class LidarEncoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super(LidarEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=3, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 5, stride=3, padding=2)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)         
        x = self.conv3(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.conv5(x)
        x = F.leaky_relu(x, inplace=True)
        x = x.view(-1, 512)
        return x
    
class NavEncoder(nn.Module):
    def __init__(self, input_dim=1, out_dim=256):
        super(NavEncoder, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, self.out_dim, 3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)         
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = F.leaky_relu(x, inplace=True)
        x = x.view(-1, self.out_dim)
        return x
    
class Generator(nn.Module):
    def __init__(self, input_dim=8, output_dim=2):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, output_dim)
        

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear2(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear3(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear4(x)
        x = torch.cos(x)
        x = self.linear5(x)
        return x
    

pixs_per_meter = 5
pixs_per_meter_heignt = 10
height = int(2*pixs_per_meter_heignt)

lidar_encoder = LidarEncoder(in_channels=height)
nav_encoder = NavEncoder(input_dim=1, out_dim=256)
decoder = Generator(input_dim=512+256+1, output_dim=2)

dummy_lidar_input = torch.randn(1, 20, 512, 512)
dummy_nav_input = torch.randn(1, 1, 200, 400)
dummy_decoder_input = torch.randn(20, 769)

t1 = time.time()
lidar_output = lidar_encoder(dummy_lidar_input)
t2 = time.time()
nav_output = nav_encoder(dummy_nav_input)
t3 = time.time()
decoder_output = decoder(dummy_decoder_input)
t4 = time.time()

print('Should Output:')
print('torch.Size([1, 512]) torch.Size([1, 256]) torch.Size([20, 2])')
print('Real Output:')
print(lidar_output.shape, nav_output.shape, decoder_output.shape)
print('Used time:')
print('lidar_encoder:', str(round(1000*(t2-t1), 1)), 'ms')
print('nav_encoder:', str(round(1000*(t3-t2), 1)), 'ms')
print('decoder:', str(round(1000*(t4-t3), 1)), 'ms')