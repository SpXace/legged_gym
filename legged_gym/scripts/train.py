# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
#import haienv
#haienv.set_env('eth')

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print(type(env))
    # print(type(env_cfg))
    # for each in env.__dir__():
    #     attr_name=each
    #     attr_value=env.__getattribute__(each)
    #     if attr_name=="obs_buf":
    #         print(attr_name,':',attr_value.shape)
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    #print("here！！！！！！！！！！！！！！！！！！！！！！！！！！")
   # lz = torch.load('/home/spxace/model_1500.pt')
    # content = torch.load('/home/spxace/anydrive_v3_lstm.pt',map_location=torch.device('cpu') )
    # print(content.state_dict())   # keys()
    # for x in content.state_dict():
    #     # print(x)
    #     print(content.state_dict()[x].shape)
    #     print("------------------------------")
    # ipp = os.getenv('MASTER_IP')
    # print('this is ip')
    # print(ipp)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    args = get_args()
    train(args)
