#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:19:57 2019

@author: hsjomaa
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from agent import deepq
import gymnasium as gym
from common.misc_util import set_global_seeds
import argparse
import logger
import numpy as np
from common.cmd_util import  make_meta_env
from common.utils import model_dir
from TabularAgents import TabularRL

def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to avoid memory allocation issues
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Optionally limit GPU usage to the first GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"Using GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPU detected, falling back to CPU.")

def main():
    configure_gpu()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='nnMeta-v40')
    parser.add_argument('--new_env_id', help='environment ID for optimization', default='Taxi-v3')  # New environment to optimize
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    parser.add_argument('--dueling', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(2e6)) #10e7
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--learning_starts', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--target_network_update_freq', type=int, default=500)
    parser.add_argument('--cell', type=int, default=32)
    parser.add_argument('--nhidden', type=int, default=128)
    parser.add_argument('--ei', default='True')
    
    args = parser.parse_args()
    args.buffer_size = args.learning_starts
    if args.env_id.startswith('nn'):
        N_t = 4; N_f = 21
    elif args.env_id.startswith('svm'):
        N_t = 7; N_f = 3
    checkpoint_path = model_dir(args)
    logger.configure(checkpoint_path)
    set_global_seeds(args.seed)
    
    # Create training environment
    training_env = make_meta_env(args.env_id, seed=args.seed)
    max_length = training_env.observation_space.shape[0]

    with tf.device('/GPU:0'):
        model = deepq.models.lstm_to_mlp(
            cell=(args.cell,N_t,N_f),
            aktiv = tf.nn.tanh,
            hiddens=[args.nhidden],
            max_length = max_length,
            dueling=bool(args.dueling),
        )

        trained_act_wrapper = deepq.learn(
            training_env,
            q_func=model,
            lr=args.lr,
            max_timesteps=args.num_timesteps,
            buffer_size=5000,
            exploration_fraction=0.8,
            exploration_final_eps=0.01,
            train_freq=args.train_freq,
            learning_starts=5000,
            target_network_update_freq=args.target_network_update_freq,
            gamma=0.99,
            prioritized_replay=bool(args.prioritized),
            prioritized_replay_alpha=args.prioritized_replay_alpha,
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_path=checkpoint_path.replace('/logs/','/checkpoints/'),
            ei=eval(args.ei),
            N_t=N_t
            )
        
    training_env.close()

if __name__ == '__main__':
    main()

