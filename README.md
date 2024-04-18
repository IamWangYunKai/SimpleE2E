# SimpleE2E
#### 安装通讯框架
```bash
git clone git@github.com:5G-Swarm/informer2.git
cd informer2
pip install .
```

#### 安装CARLA
下载并安装[CARLA 0.9.9.4](https://github.com/carla-simulator/carla/releases/tag/0.9.9)

#### SSH连接
```bash
ssh root@[ip_address]
输入密码：xxxxx
```

#### 如果硬盘没挂载

```bash
mount /dev/sda /data
df -h
```

#### Step1：板子启动（板子）
需要安装Ascend-cann-toolkit和Ascend-cann-nnrt的7.0.RC1版本
* 进入文件夹
```bash
cd [your_path]/Atlas/
```
* source环境
 ```bash
source /data/ascend-toolkit/set_env.sh
source /data/Ascend/nnrt/set_env.sh
 ```

* 测试模型是否可用
 ```bash
python3 test_dummy_model.py
 ```

#### Step2：CARLA准备（主机）

* 进入文件夹
```bash
cd ~/CARLA_0.9.9.4
```
* 启动CARLA
```bash
DISPLAY= ./CarlaUE4.sh -opengl -quality-level=Low #（建议）无窗口，低质量
```
* 进入程序文件夹
```bash
cd [your_path]/scripts
```

#### Step3：运行系统
* 先启动板子的程序
```bash
python3 run_client.py
```

* 修改config.yaml中板子的IP
```bash
target_info:
    [ip_address]
```

* 再启动主机的程序
```bash
python run_server.py
```

* 如果要本地运行不借助板子
```bash
python run_local.py
```

* 加障碍物（新开窗口）
```bash
cd ~/CARLA_0.9.9.4/PythonAPI/examples
python spawn_npc.py -n 20 -w 0 # 20车 0行人
```

[![](./gif/video.gif)](https://www.bilibili.com/video/BV1er421V7s2/?share_source=copy_web&vd_source=fd36c231f4e04f7f0c26e0320abcaccc)


#### 引用
```
Wang Y, Zhang D, Wang J, et al. Imitation learning of hierarchical driving model: from continuous intention to continuous trajectory[J]. IEEE Robotics and Automation Letters, 2021, 6(2): 2477-2484.
```
或者
```
@article{wang2021imitation,
  title={Imitation learning of hierarchical driving model: from continuous intention to continuous trajectory},
  author={Wang, Yunkai and Zhang, Dongkun and Wang, Jingke and Chen, Zexi and Li, Yuehua and Wang, Yue and Xiong, Rong},
  journal={IEEE Robotics and Automation Letters},
  volume={6},
  number={2},
  pages={2477--2484},
  year={2021},
  publisher={IEEE}
}
```