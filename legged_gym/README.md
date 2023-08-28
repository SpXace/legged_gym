# Legged Gym

## Environment
1. 创建conda环境

```shell
conda create --name env_name python=3.8
```

2. 安装pytorch 1.10 with cuda-11.3

```shell
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f
```
3. pypi

```shell
pip install numpy==1.19.5
pip install matplotlib==3.5.0
pip install setuptools==59.5.0
pip install tensorboard
```

3. 下载Isaac Gym

```shell
git clone https://github.com/SpXace/isaacgym
cd isaacgym/python && pip install -e .
```

4. 下载rsl_rl

```shell
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl && pip install -e .
```

5. 代码 & 下载legged gym

```shell
git clone https://github.com/SpXace/legged_gym

cd legged_gym && pip install -e .
```

## Train
```shell
python legged_gym/scripts/train.py --task=a1 --rl_device="cuda:3"
```

