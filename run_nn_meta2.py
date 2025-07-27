from agent import ppo
import gymnasium as gym
from common.misc_util import set_global_seeds
import argparse
import logger
import numpy as np
from common.cmd_util import make_meta_env
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from common.utils import model_dir
import wandb

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
    wandb.init(project="RL-HPO-Visualization2")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='nnMeta-v40')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(17e5))  # 10e7
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--buffer_size', type=int, default=2048)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--train_iters', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--cell', type=int, default=32)
    parser.add_argument('--nhidden', type=int, default=128)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--entropy_coef', type=int, default=0.1)

    args = parser.parse_args()

    # args.lr = wandb.config.get("lr", args.lr)
    # args.batch_size = wandb.config.get("batch_size", args.batch_size)
    # args.train_epochs = wandb.config.get("train_epochs", args.train_epochs)
    # args.clip_ratio = wandb.config.get("clip_ratio", args.clip_ratio)
    # args.entropy_coef = wandb.config.get("entropy_coef", args.entropy_coef)

    # Set up environment-specific parameters
    if args.env_id.startswith('nn'):
        N_t = 4
        N_f = 21
    elif args.env_id.startswith('svm'):
        N_t = 7
        N_f = 3

    # Configure logging and checkpoints
    checkpoint_path = model_dir(args)
    logger.configure(checkpoint_path)
    set_global_seeds(args.seed)

    # Create environment
    env = make_meta_env(args.env_id, seed=args.seed)
    max_length = env.observation_space.shape[0]

    # Create model networks
    with tf.device('/GPU:0'):
            actor_network, value_network = ppo.lstm_to_mlp(
            cell=(args.cell, N_t, N_f),
            aktiv=tf.nn.tanh,
            hiddens=[args.nhidden],
            max_length=max_length,
        )

        # Train PPO agent
            trained_act_wrapper = ppo.learn(
            env=env,
            actor_network=actor_network,
            value_network=value_network,
            max_lives = 1,
            lr = args.lr,
            max_timesteps=args.num_timesteps,
            buffer_size=args.buffer_size,
            train_freq=1,
            batch_size=args.batch_size,
            print_freq = 100,
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_path=checkpoint_path.replace('/logs/', '/checkpoints/'),
            learning_starts = 1000,
            gamma = args.gamma,
            lam = args.lam,
            clip_ratio = args.clip_ratio,
            train_epochs = 10,
            N_t=N_t,
            callback = None,
            entropy_coef=args.entropy_coef
        )

    env.close()


if __name__ == '__main__':
    main()