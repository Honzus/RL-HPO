# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import numpy as np
# import common.tf_util as U
# from agent.deepq.simple import ActWrapper
# import os
# import wandb
# import tempfile
# import logger
# from agent.deepq.utils import ObservationInput
# from common.tf_util import load_state, save_state
# import copy

# def preprocess_observation(obs):
#     obs = np.array(obs, dtype=np.float32)
#     orig_shape = obs.shape

#     # Flatten to 2D if needed (for easy processing)
#     obs_flat = obs.reshape(-1, obs.shape[-1])

#     main_obs = obs_flat[:, :4]
#     metafeatures = obs_flat[:, 4:]

#     metafeatures[metafeatures == -999999] = -1.0
#     metafeatures[metafeatures == 999999] = 1.0

#     metafeatures[metafeatures == 1000] = 2 * (1000 + 999999) / (999999 + 999999) - 1

#     metafeatures[:, 4] = 2 * metafeatures[:, 4] / 500 - 1
#     metafeatures[:, 0] = 2 * (metafeatures[:, 0]-1)/(99-1) - 1
#     metafeatures[:, 2] = 2 * (metafeatures[:, 2]-3) /(99-3) - 1


#     # Concatenate back
#     processed = np.concatenate([main_obs, metafeatures], axis=1)

#     # Restore original shape
#     processed = processed.reshape(orig_shape)
#     return processed

# def scope_vars(scope, trainable_only=False):
#     """Get variables inside a scope
#     Parameters
#     ----------
#     scope: str or VariableScope
#         scope in which the variables reside.
#     trainable_only: bool
#         whether or not to return only the variables that were marked as trainable.
#     Returns
#     -------
#     vars: [tf.Variable]
#         list of variables in `scope`.
#     """
#     return tf.get_collection(
#         tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
#         scope=scope if isinstance(scope, str) else scope.name
#     )


# def lstm_to_mlp(cell, hiddens, max_length, aktiv=tf.nn.tanh, dueling=False):
#     """
#     Creates an MLP from an LSTM embedding for PPO

#     Parameters
#     ----------
#     cell: tuple
#         cell[0] is the number of LSTM units
#         cell[1] is the number of time steps (N_t)
#         cell[2] is the number of features (N_f)
#     hiddens: list
#         List of hidden layer sizes
#     max_length: int
#         Maximum sequence length
#     aktiv: tf activation function
#         Activation function
#     dueling: bool
#         Whether to use dueling architecture

#     Returns
#     -------
#     q_func: function
#         q network model function
#     """
#     n_lstm, n_t, n_f = cell

#     def actor_network(x, seq, num_actions, scope, reuse=False):
#         with tf.variable_scope(scope, reuse=reuse):
#             batch_size = tf.shape(x)[0]

#             # Reshape for LSTM
#             # Assuming x has shape [batch_size, max_length, n_f]
#             # First extract the metafeatures (assumed to be at the beginning of each sequence)
#             metafeatures = x[:, 0, -n_f:]  # Shape: [batch_size, n_f]

#             # LSTM processing
#             lstm_cell = tf.nn.rnn_cell.LSTMCell(n_lstm, activation=aktiv)
#             lstm_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

#             # Process steps up to the current sequence position for each sample in batch
#             lstm_outputs = []
#             for t in range(n_t):
#                 # Extract features for this timestep (ignoring metafeatures)
#                 # Create a boolean mask of shape [batch_size]
#                 mask = tf.less_equal(t, seq)

#                 # Reshape mask to [batch_size, 1] for proper broadcasting
#                 mask_expanded = tf.expand_dims(mask, axis=1)

#                 # Tile mask to match feature dimensions [batch_size, n_f]
#                 mask_tiled = tf.tile(mask_expanded, [1, n_f])

#                 # Apply mask
#                 zeros = tf.zeros([batch_size, n_f], dtype=tf.float32)
#                 t_features = tf.where(mask_tiled, x[:, t, :n_f], zeros)

#                 # Process through LSTM
#                 output, lstm_state = lstm_cell(t_features, lstm_state)
#                 lstm_outputs.append(output)

#             # Stack all outputs [n_t, batch_size, n_lstm]
#             lstm_outputs_stacked = tf.stack(lstm_outputs)  # Shape: [n_t, batch_size, n_lstm]

#             # Transpose to [batch_size, n_t, n_lstm]
#             lstm_outputs_transposed = tf.transpose(lstm_outputs_stacked, [1, 0, 2])

#             # Gather the correct output for each sample in batch based on seq
#             batch_indices = tf.range(batch_size)
#             # Clamp seq values to be within valid range
#             safe_seq = tf.minimum(seq, n_t - 1)
#             indices = tf.stack([batch_indices, safe_seq], axis=1)
#             lstm_output = tf.gather_nd(lstm_outputs_transposed, indices)

#             # Combine LSTM output with metafeatures
#             combined = tf.concat([lstm_output, metafeatures], axis=1)

#             # Process through hidden layers
#             hidden = combined
#             for h in hiddens:
#                 hidden = tf.layers.dense(hidden, h, activation=aktiv)

#             # Output layer (logits)
#             logits = tf.layers.dense(hidden, num_actions)
#             logits = tf.clip_by_value(logits, -1000.0, 1000.0)  # Clip to prevent NaNs

#             return logits

#     def value_network(x, seq, scope, reuse=False):
#         with tf.variable_scope(scope, reuse=reuse):
#             batch_size = tf.shape(x)[0]

#             # Reshape for LSTM
#             # Assuming x has shape [batch_size, max_length, n_f]
#             # First extract the metafeatures (assumed to be at the beginning of each sequence)
#             metafeatures = x[:, 0, -n_f:]  # Shape: [batch_size, n_f]

#             # LSTM processing
#             lstm_cell = tf.nn.rnn_cell.LSTMCell(n_lstm, activation=aktiv)
#             lstm_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

#             # Process steps up to the current sequence position for each sample in batch
#             lstm_outputs = []
#             for t in range(n_t):
#                 # Extract features for this timestep (ignoring metafeatures)
#                 # Create a boolean mask of shape [batch_size]
#                 mask = tf.less_equal(t, seq)

#                 # Reshape mask to [batch_size, 1] for proper broadcasting
#                 mask_expanded = tf.expand_dims(mask, axis=1)

#                 # Tile mask to match feature dimensions [batch_size, n_f]
#                 mask_tiled = tf.tile(mask_expanded, [1, n_f])

#                 # Apply mask
#                 zeros = tf.zeros([batch_size, n_f], dtype=tf.float32)
#                 t_features = tf.where(mask_tiled, x[:, t, :n_f], zeros)

#                 # Process through LSTM
#                 output, lstm_state = lstm_cell(t_features, lstm_state)
#                 lstm_outputs.append(output)

#             # Stack all outputs [n_t, batch_size, n_lstm]
#             lstm_outputs_stacked = tf.stack(lstm_outputs)  # Shape: [n_t, batch_size, n_lstm]

#             # Transpose to [batch_size, n_t, n_lstm]
#             lstm_outputs_transposed = tf.transpose(lstm_outputs_stacked, [1, 0, 2])

#             # Gather the correct output for each sample in batch based on seq
#             batch_indices = tf.range(batch_size)
#             # Clamp seq values to be within valid range
#             safe_seq = tf.minimum(seq, n_t - 1)
#             indices = tf.stack([batch_indices, safe_seq], axis=1)
#             lstm_output = tf.gather_nd(lstm_outputs_transposed, indices)

#             # Combine LSTM output with metafeatures
#             combined = tf.concat([lstm_output, metafeatures], axis=1)

#             # print("LSTM output stats:", tf.reduce_mean(lstm_output), tf.math.reduce_std(lstm_output))
#             # print("Combined features stats:", tf.reduce_mean(combined), tf.math.reduce_std(combined))

#             # Process through hidden layers
#             hidden = combined
#             for h in hiddens:
#                 hidden = tf.layers.dense(hidden, h, activation=aktiv)

#             # Output layer (value)
#             value = tf.layers.dense(hidden, 1)
#             value = tf.clip_by_value(value, -1000.0, 1000.0)  # Clip to prevent NaNs
#             return value

#     return actor_network, value_network


# def build_policy(make_obs_ph, actor_network, value_network, num_actions, scope="ppo", entropy_coef=0.1, reuse=None):
#     """
#     Build PPO policy and value networks

#     Parameters
#     ----------
#     make_obs_ph: function
#         function that creates observation placeholder
#     actor_network: function
#         function that takes observation tensor and returns action logits
#     value_network: function
#         function that takes observation tensor and returns value estimate
#     num_actions: int
#         number of possible actions
#     scope: str
#         name of the scope
#     reuse: bool
#         whether or not to reuse existing scope

#     Returns
#     -------
#     act: function
#         function to select and return actions
#     train: function
#         function to train the PPO policy
#     update_old_policy: function
#         function to update target policy for stable updates
#     """
#     with tf.variable_scope(scope, reuse=reuse):
#         # Placeholders
#         observations_ph = make_obs_ph("observation")
#         seq_t_ph = tf.placeholder(tf.int32, [None], name="seq_t")
#         actions_ph = tf.placeholder(tf.int32, [None], name="actions")
#         advantages_ph = tf.placeholder(tf.float32, [None], name="advantages")
#         returns_ph = tf.placeholder(tf.float32, [None], name="returns")
#         old_neglogp_ph = tf.placeholder(tf.float32, [None], name="old_neglogp")
#         old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred")
#         lr_ph = tf.placeholder(tf.float32, [], name="learning_rate")
#         clip_param_ph = tf.placeholder(tf.float32, [], name="clip_param")

#         temperature_ph = tf.placeholder_with_default(1.0, shape=(), name="temperature")  # Softmax temperature

#         # Policy network (Actor)
#         with tf.variable_scope("policy"):
#             policy_logits = actor_network(observations_ph.get(), seq_t_ph, num_actions, scope="policy_net")

#             # Stabilize softmax
#             scaled_logits = policy_logits / temperature_ph
#             stable_logits = policy_logits - tf.reduce_max(scaled_logits, axis=1, keepdims=True)
#             policy_probs = tf.nn.softmax(stable_logits)

#             policy_dist = tf.distributions.Categorical(probs=policy_probs)
#             action = policy_dist.sample()  # Rely on policy's stochasticity
#             action_source = tf.ones_like(action, dtype=tf.int32) * 2  # Always policy

#             # Stable categorical distribution for log probability
#             stable_probs = policy_probs + 1e-10
#             stable_probs = stable_probs / tf.reduce_sum(stable_probs, axis=1, keepdims=True)
#             policy_dist = tf.distributions.Categorical(probs=stable_probs)
#             neglogp = -policy_dist.log_prob(action)
#             neglogp_actions = -policy_dist.log_prob(actions_ph)

#         # Value network (Critic)
#         with tf.variable_scope("value"):
#             value_pred = value_network(observations_ph.get(), seq_t_ph, scope="value_net")
#             value_pred = tf.squeeze(value_pred, axis=1)

#         # Old (target) policy, used for KL and ratio calculation
#         with tf.variable_scope("old_policy"):
#             old_policy_logits = actor_network(observations_ph.get(), seq_t_ph, num_actions, scope="old_policy_net")
#             old_policy_probs = tf.nn.softmax(old_policy_logits)
#             old_policy_dist = tf.distributions.Categorical(probs=old_policy_probs)
#             old_neglogp = -old_policy_dist.log_prob(actions_ph)

#         # Get trainable variables for all networks
#         policy_vars = scope_vars(scope + "/policy", trainable_only=True)
#         value_vars = scope_vars(scope + "/value", trainable_only=True)
#         old_policy_vars = scope_vars(scope + "/old_policy", trainable_only=True)

#         # print("Value network variables:")
#         # for var in value_vars:
#         #     print(var.name)

#         # Loss calculations
#         # Policy loss with clipping
#         ratio = tf.exp(
#             old_neglogp_ph - neglogp_actions)  # ratio between old and new policy, should be one at the first iteration
#         clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_param_ph, 1.0 + clip_param_ph)
#         policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages_ph, clipped_ratio * advantages_ph))

#         # Value loss
#         value_clipped = old_vpred_ph + tf.clip_by_value(value_pred - old_vpred_ph, -clip_param_ph, clip_param_ph)
#         value_loss1 = tf.square(value_pred - returns_ph)
#         value_loss2 = tf.square(value_clipped - returns_ph)
#         value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss1, value_loss2))

#         # Entropy loss to encourage exploration
#         entropy = tf.reduce_mean(policy_dist.entropy())

#         # Total loss
#         loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

#         # KL divergence for adaptive learning rate
#         approxkl = tf.reduce_mean(old_neglogp_ph - neglogp_actions)

#         # Create optimizer
#         optimizer = tf.train.AdamOptimizer(learning_rate=lr_ph)

#         # Gradient clipping
#         grads_and_vars = optimizer.compute_gradients(loss, var_list=policy_vars + value_vars)
#         grads, vars = zip(*grads_and_vars)
#         # print("Variables in optimizer:")
#         # for grad, var in grads_and_vars:
#         #     print(var.name)
#         grads_clipped, _ = tf.clip_by_global_norm(grads, 1)
#         optimize_expr = optimizer.apply_gradients(list(zip(grads_clipped, vars)))
#         # For debugging: get the norm of each gradient
#         grad_norms = [tf.norm(g) if g is not None else tf.constant(0.0) for g in grads_clipped]


#         # Update old policy
#         update_old_policy_expr = []
#         for var, var_old in zip(sorted(policy_vars, key=lambda v: v.name),
#                                 sorted(old_policy_vars, key=lambda v: v.name)):
#             update_old_policy_expr.append(var_old.assign(var))
#         update_old_policy_expr = tf.group(*update_old_policy_expr)

#         # Create callable functions
#         def act(obs, seq):
#             """Take an observation and return an action and its value"""
#             a, v, nlp = sess.run([action, value_pred, neglogp],
#                                  feed_dict={observations_ph.get(): obs, seq_t_ph: seq})
#             return a, v, nlp

#         def compute_values(obs, seq):
#             """Compute value estimates for a batch of observations"""
#             return sess.run(value_pred,
#                             feed_dict={observations_ph.get(): obs, seq_t_ph: seq})

#         def train(lr, clip_param, obs, seq, actions, values, returns, neglogpacs, advantages):
#             """Update policy and value networks"""
#             td_map = {
#                 lr_ph: lr,
#                 clip_param_ph: clip_param,
#                 observations_ph.get(): obs,
#                 seq_t_ph: seq,
#                 actions_ph: actions,
#                 advantages_ph: advantages,
#                 returns_ph: returns,
#                 old_neglogp_ph: neglogpacs,
#                 old_vpred_ph: values
#             }

#             policy_loss_val, value_loss_val, entropy_val, approxkl_val, grad_norms_val, _ = sess.run(
#                 [policy_loss, value_loss, entropy, approxkl, grad_norms, optimize_expr],
#                 feed_dict=td_map
#             )

#             # print("Value loss:", value_loss_val)
#             # for grad, var in zip(grad_norms_val, vars):
#             #     if grad is not None:
#             #         print(var.name, "grad norm:", np.linalg.norm(grad))
#             #     else:
#             #         print(var.name, "grad is None")

#             policy_probs_val = tf.get_default_session().run(policy_probs, feed_dict=td_map)

#             return policy_loss_val, value_loss_val, entropy_val, approxkl_val

#         def update_old_policy():
#             """Update target policy to current policy for stable updates"""
#             return sess.run(update_old_policy_expr)

#         # Save session reference
#         sess = tf.get_default_session()

#     return act, train, update_old_policy, compute_values


# class PPOBuffer:
#     """Buffer for storing trajectory data for PPO training with flexible capacity"""

#     def __init__(self, size, obs_shape, gamma=0.99, lam=0.95):
#         self.obs = np.zeros((size,) + obs_shape, dtype=np.float32)
#         self.seq = np.zeros(size, dtype=np.int32)
#         self.actions = np.zeros(size, dtype=np.int32)
#         self.rewards = np.zeros(size, dtype=np.float32)
#         self.values = np.zeros(size, dtype=np.float32)
#         self.neglogpacs = np.zeros(size, dtype=np.float32)
#         self.dones = np.zeros(size, dtype=np.float32)
#         self.advantages = np.zeros(size, dtype=np.float32)
#         self.returns = np.zeros(size, dtype=np.float32)

#         self.gamma = gamma
#         self.lam = lam
#         self.ptr, self.path_start_idx = 0, 0
#         self.max_size = size
#         self._next_idx = 0
#         self._full = False

#     def store(self, obs, seq, action, reward, value, neglogpac, done):
#         """Store a single transition"""
#         assert self.ptr < self.max_size
#         self.obs[self.ptr] = obs
#         self.seq[self.ptr] = seq
#         self.actions[self.ptr] = action
#         self.rewards[self.ptr] = reward
#         self.values[self.ptr] = value
#         self.neglogpacs[self.ptr] = neglogpac
#         self.dones[self.ptr] = done
#         self.ptr += 1

#         # Update _next_idx and check if buffer is full
#         self._next_idx = self.ptr % self.max_size
#         if self.ptr >= self.max_size:
#             self._full = True

#     def finish_path(self, last_val=0):
#         """
#         Compute advantages and returns for a completed trajectory
#         using GAE (Generalized Advantage Estimation)
#         """
#         path_slice = slice(self.path_start_idx, self.ptr)
#         path_len = self.ptr - self.path_start_idx

#         # Handle empty paths
#         if path_len <= 0:
#             self.path_start_idx = self.ptr
#             return

#         rewards = np.append(self.rewards[path_slice], last_val)
#         values = np.append(self.values[path_slice], last_val)
#         dones = np.append(self.dones[path_slice], 0)

#         # Safe GAE calculation - prevent NaN propagation
#         deltas = rewards[:-1] + self.gamma * values[1:] * (1 - dones[:-1]) - values[:-1]

#         # Clip extreme values
#         deltas = np.clip(deltas, -100.0, 100.0)

#         # Calculate advantages with safety
#         gae = 0
#         advantages = np.zeros(path_len)

#         for t in reversed(range(path_len)):
#             gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
#             # Prevent extreme values
#             gae = np.clip(gae, -100.0, 100.0)
#             advantages[t] = gae

#         # Store advantages and compute returns
#         self.advantages[path_slice] = advantages
#         self.returns[path_slice] = advantages + self.values[path_slice]

#         # Clip returns to prevent extreme values
#         self.returns[path_slice] = np.clip(self.returns[path_slice], -10.0, 10.0)

#         self.path_start_idx = self.ptr

#     def get(self):
#         """Get stored data for training, using available data"""
#         # Use available data even if buffer is not completely full
#         available_size = self.ptr if not self._full else self.max_size

#         if available_size < 1:
#             raise ValueError("No data available in buffer")

#         # If buffer wrapped around, use data from 0 to _next_idx
#         if self._full:
#             indices = np.arange(available_size)
#         else:
#             indices = np.arange(0, self.ptr)

#         # Normalize advantages
#         if np.any(np.isnan(self.advantages[indices])):
#             self.advantages[indices] = np.nan_to_num(self.advantages[indices])

#             # Safe advantage normalization
#         if len(indices) > 1:  # Only normalize if we have multiple samples
#             adv_mean = np.mean(self.advantages[indices])
#             adv_std = np.std(self.advantages[indices])

#             # Prevent division by zero
#             if adv_std < 1e-8:
#                 adv_std = 1.0

#             self.advantages[indices] = (self.advantages[indices] - adv_mean) / adv_std

#             # Final safety check for NaNs
#         for array in [self.obs[indices], self.seq[indices], self.actions[indices],
#                       self.values[indices], self.returns[indices],
#                       self.neglogpacs[indices], self.advantages[indices]]:
#             if np.any(np.isnan(array)):
#                 # Replace NaNs with safe values
#                 np.nan_to_num(array, copy=False)

#         # Prepare data for training
#         return (
#             self.obs[indices],
#             self.seq[indices],
#             self.actions[indices],
#             self.values[indices],
#             self.returns[indices],
#             self.neglogpacs[indices],
#             self.advantages[indices]
#         )

#     def clear(self):
#         """Reset buffer"""
#         self.ptr, self.path_start_idx = 0, 0
#         self._full = False

#     def __len__(self):
#         """Return current buffer size"""
#         return self.ptr if not self._full else self.max_size


# def learn(env,
#           actor_network,
#           value_network,
#           max_lives=1,
#           lr=3e-4,
#           max_timesteps=int(1e6),
#           buffer_size=2048,
#           train_freq=1,
#           batch_size=64,
#           print_freq=100,
#           checkpoint_freq=10000,
#           checkpoint_path=None,
#           learning_starts=1000,
#           gamma=0.99,
#           lam=0.95,
#           clip_ratio=0.2,
#           train_epochs=10,
#           N_t=4,
#           callback=None,
#           entropy_coef=0.1):
#     """
#     Train a PPO agent on the HPO environment

#     Parameters
#     ----------
#     env: gym.Env
#         Environment to train on
#     actor_network: function
#         Network that takes observations and returns action logits
#     value_network: function
#         Network that takes observations and returns value estimates
#     max_lives: int
#         Maximum number of lives before switching dataset
#     lr: float
#         Learning rate
#     max_timesteps: int
#         Total number of timesteps to train for
#     buffer_size: int
#         Size of the replay buffer
#     train_freq: int
#         How often to run PPO updates (in steps)
#     batch_size: int
#         Size of minibatches for training
#     print_freq: int
#         How often to print progress information
#     checkpoint_freq: int
#         How often to save model checkpoints (timesteps)
#     checkpoint_path: str
#         Path to save model checkpoints
#     learning_starts: int
#         How many steps to collect before starting training
#     gamma: float
#         Discount factor
#     lam: float
#         GAE lambda parameter
#     clip_ratio: float
#         PPO clipping parameter
#     train_epochs: int
#         Number of PPO epochs per update
#     N_t: int
#         Number of hyperparameter selection steps
#     callback: function
#         Optional callback function to monitor training

#     Returns
#     -------
#     act: ActWrapper
#         Wrapper over act function
#     """
#     # Create all the functions necessary to train the model
#     sess = tf.Session()
#     sess.__enter__()

#     # Create observation placeholder
#     def make_obs_ph(name):
#         return ObservationInput(env.observation_space, name=name)

#     # Get observation space shape and action space size
#     obs_shape = env.observation_space.shape
#     num_actions = env.action_space.n

#     # Build policy networks and training functions
#     act, train, update_old_policy, compute_values = build_policy(
#         make_obs_ph=make_obs_ph,
#         actor_network=actor_network,
#         value_network=value_network,
#         num_actions=num_actions,
#         entropy_coef=entropy_coef
#         # optimizer=tf.train.AdamOptimizer(learning_rate=lr),
#         # session=sess
#     )

#     # Store act parameters for saving/loading
#     act_params = {
#         'make_obs_ph': make_obs_ph,
#         'actor_network': actor_network,
#         'value_network': value_network,
#         'num_actions': num_actions,
#         'N_t': N_t
#     }

#     # Create ActWrapper that can be saved/loaded
#     act = ActWrapper(act, act_params)

#     # Initialize variables
#     U.initialize()
#     update_old_policy()  # Sync old policy with new policy

#     # Create buffer for storing experience
#     buffer = PPOBuffer(buffer_size, obs_shape, gamma, lam)

#     # Initialize tracking variables
#     episode_rewards = [0.0]
#     episode_ei = [0.0]  # Including ei tracking for consistency
#     episode_lengths = [1]
#     policy_losses = []
#     value_losses = []
#     entropies = []
#     kls = []
#     saved_mean_reward = None

#     # Initialize dataset tracking as in DeepQ implementation
#     dataset_idx = 0
#     dataset_ctr = np.zeros(shape=(len(env.env.ale.metadata)))
#     dataset_ctr[dataset_idx] += 1

#     # Initialize environment state using the same pattern as DeepQ
#     state = env.reset()
#     obs = np.zeros(shape=env.observation_space.shape)
#     obs[0, :] = np.append(np.repeat(np.NaN, repeats=N_t), env.env.ale.metadata[dataset_idx]['features']).reshape(1, -1)
#     obs = np.nan_to_num(obs, copy=True, nan=0.0)  # Replace NaN with 0
#     obs = preprocess_observation(obs)  # ADD THIS LINE
#     reset = True
#     prev_r = 0

#     wandb.init(project="RL-HPO-Visualization",
#                name=f"PPO-{env.env_id if hasattr(env, 'env_id') else 'unknown'}",
#                config={
#                    "algorithm": "PPO",
#                    "env_id": env.env_id if hasattr(env, 'env_id') else 'unknown',
#                    "learning_rate": lr,
#                    "buffer_size": buffer_size,
#                    "batch_size": batch_size,
#                    "gamma": gamma,
#                    "lam": lam,
#                    "clip_ratio": clip_ratio,
#                    "train_epochs": train_epochs
#                })

#     wandb.log({"policy_loss_mean_100": 0,
#                "value_loss_mean_100": 0,
#                "entropy_mean_100": 0,
#                "approx_kl_mean_100": 0,
#                "episode_reward": 0,
#                "episode_length": 1,
#                "mean_100ep_reward": 0,
#                "episode_length_mean_100": 0,
#                "timesteps": 0
#                })

#     def wandb_callback(locals, globals):
#         # Log metrics at the end of each episode
#         if len(locals['episode_rewards']) > 0:  # Log every 100 timesteps

#             last_100_rewards = locals['episode_rewards'][-100:]
#             last_100_lengths = locals['episode_lengths'][-100:]
#             last_100_policy_losses = locals.get('policy_losses', [0])[-100:]
#             last_100_value_losses = locals.get('value_losses', [0])[-100:]
#             last_100_entropies = locals.get('entropies', [0])[-100:]
#             last_100_kls = locals.get('kls', [0])[-100:]

#             wandb.log({
#                 "episode_reward": locals['episode_rewards'][-1],
#                 "episode_length": locals['episode_lengths'][-1],
#                 "mean_100ep_reward": np.mean(last_100_rewards),
#                 "episode_length_mean_100": np.mean(last_100_lengths),
#                 "policy_loss_mean_100": np.mean(last_100_policy_losses),
#                 "value_loss_mean_100": np.mean(last_100_value_losses),
#                 "entropy_mean_100": np.mean(last_100_entropies),
#                 "approx_kl_mean_100": np.mean(last_100_kls),
#                 "timesteps": locals['t']
#             })
#         return False

#     # Combine the original callback with wandb callback
#     def combined_callback(locals, globals):
#         if callback is not None:
#             callback_result = callback(locals, globals)
#             if callback_result:
#                 return True
#         return wandb_callback(locals, globals)

#     # Save/load model handling
#     with tempfile.TemporaryDirectory() as td:
#         td = checkpoint_path or td
#         model_file = os.path.join(td, "model")
#         model_saved = False

#         if tf.train.latest_checkpoint(td) is not None:
#             load_state(model_file)
#             logger.log('Loaded model from {}'.format(model_file))
#             model_saved = True

#         # Main training loop
#         for t in range(max_timesteps):

#             # Get current sequence position for observation - match DeepQ's pattern
#             seq = env.env._get_ep_len()

#             # Get action from policy - match DeepQ's pattern of passing sequence
#             action, value, neglogp = act(np.array(obs)[None], [seq])
#             action = action[0]  # Extract scalar from array
#             value = value[0]
#             neglogp = neglogp[0]

#             # Take step in environment using DeepQ's pattern
#             new_state, rew, done, _ = env.step(action.item(), dataset_id=dataset_idx)

#             # Update observation using DeepQ's pattern
#             new_obs = copy.copy(obs)
#             new_obs[env.env._get_ep_len(), :] = np.append(new_state, np.append(rew, env.env.ale.metadata[dataset_idx][
#                 'features'])).reshape(1, -1)
#             new_obs = np.nan_to_num(new_obs, copy=True, nan=0.0)  # Replace NaN with 0
#             new_obs = preprocess_observation(new_obs)  # preprocess the observation

#             if np.any(np.abs(new_obs) > 1e5):
#                 print("Warning: Extreme value in observation!", new_obs)

#             # Store in PPO buffer
#             buffer.store(obs, seq, action, rew, value, neglogp, float(done))

#             # Update tracking variables
#             obs = new_obs
#             episode_rewards[-1] += rew
#             prev_r = copy.copy(rew)

#             # Handle episode termination - match DeepQ's reset pattern
#             if done:
#                 episode_lengths[-1] = env.env._get_ep_len()

#                 # Finish path for PPO calculations
#                 buffer.finish_path()

#                 # Reset state using DeepQ's pattern
#                 state = env.reset()
#                 obs = np.zeros(shape=env.observation_space.shape)
#                 obs[0, :] = np.append(np.repeat(np.NaN, repeats=N_t),
#                                       env.env.ale.metadata[dataset_idx]['features']).reshape(1, -1)
#                 obs = np.nan_to_num(obs, copy=True, nan=0.0)

#                 # Update dataset tracking as in DeepQ
#                 env.env.ale._used_lives += 1
#                 if env.env.ale._used_lives % max_lives == 0:
#                     dataset_idx += 1
#                     dataset_idx = dataset_idx % len(env.env.ale.metadata)
#                     dataset_ctr[dataset_idx] += 1

#                 # Start new episode
#                 episode_lengths.append(1)
#                 episode_rewards.append(0.0)
#                 episode_ei.append(0.0)
#                 reset = True
#                 prev_r = 0

#             # Check if we need to run PPO updates (similar to DeepQ training check)
#             if t > learning_starts and t % train_freq == 0 and len(buffer) >= batch_size:
#                 # Get data from buffer
#                 obs_buffer, seq_buffer, act_buffer, val_buffer, ret_buffer, old_neglogp_buffer, adv_buffer = buffer.get()

#                 obs_buffer = preprocess_observation(obs_buffer)

#                 # Tracking stats
#                 avg_policy_loss = 0
#                 avg_value_loss = 0
#                 sum_entropy = 0.0
#                 n_updates = 0
#                 avg_approxkl = 0

#                 policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ppo/value")
#                 weights_before = tf.get_default_session().run(policy_vars)

#                 # Run PPO updates for multiple epochs
#                 for _ in range(train_epochs):
#                     # Shuffle data
#                     buffer_size = len(buffer)
#                     indices = np.arange(buffer_size)
#                     np.random.shuffle(indices)

#                     # Train in mini-batches
#                     for start in range(0, buffer_size, batch_size):
#                         end = start + batch_size
#                         batch_indices = indices[start:end]
#                         if len(batch_indices) < batch_size:  # Skip small last batch
#                             continue

#                         # print("Value network input batch stats:",
#                         #       "mean", np.mean(obs_buffer[batch_indices]),
#                         #       "std", np.std(obs_buffer[batch_indices]),
#                         #       "min", np.min(obs_buffer[batch_indices]),
#                         #       "max", np.max(obs_buffer[batch_indices]))

#                         # Update policy and value networks
#                         policy_loss, value_loss, entropy, approxkl = train(
#                             lr=lr,
#                             clip_param=clip_ratio,
#                             obs=obs_buffer[batch_indices],
#                             seq=seq_buffer[batch_indices],
#                             actions=act_buffer[batch_indices],
#                             advantages=adv_buffer[batch_indices],
#                             returns=ret_buffer[batch_indices],
#                             neglogpacs=old_neglogp_buffer[batch_indices],
#                             values=val_buffer[batch_indices]
#                         )

#                         # Track stats
#                         avg_policy_loss += policy_loss / train_epochs
#                         avg_value_loss += value_loss / train_epochs
#                         sum_entropy += entropy
#                         n_updates += 1
#                         avg_approxkl += approxkl / train_epochs

#                 # Update old policy after all updates
#                 update_old_policy()

#                 weights_after = tf.get_default_session().run(policy_vars)
#                 # for w_b, w_a in zip(weights_before, weights_after):
#                 #     print("Policy weight diff:", np.sum(np.abs(w_b - w_a)))

#                 # Store stats for logging
#                 policy_losses.append(avg_policy_loss)
#                 value_losses.append(avg_value_loss)
#                 mean_entropy = sum_entropy / max(1, n_updates)
#                 entropies.append(mean_entropy)
#                 kls.append(avg_approxkl)

#                 if combined_callback is not None:
#                     if combined_callback(locals(), globals()):
#                         break

#                 # Clear buffer after updates
#                 buffer.clear()

#             # Print progress info - match DeepQ's logging pattern
#             mean_100ep_length = round(
#                 np.mean(episode_lengths[-101:-1]) if len(episode_lengths) > 101 else np.mean(episode_lengths), 4)
#             mean_100ep_reward = round(
#                 np.mean(episode_rewards[-101:-1]) if len(episode_rewards) > 101 else np.mean(episode_rewards), 4)
#             mean_100ep_reward_running = round(
#                 np.mean(episode_rewards[-201:-100]) if len(episode_rewards) > 201 else np.mean(episode_rewards), 4)
#             num_episodes = len(episode_rewards)

#             if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
#                 logger.record_tabular("policy_loss", round(np.mean(policy_losses), 4) if policy_losses else 0)
#                 logger.record_tabular("value_loss", round(np.mean(value_losses), 4) if value_losses else 0)
#                 logger.record_tabular("entropy", round(np.mean(entropies), 4) if entropies else 0)
#                 logger.record_tabular("approx_kl", round(np.mean(kls), 4) if kls else 0)
#                 logger.record_tabular("steps", t)
#                 logger.record_tabular("most_used_dataset", np.argmax(dataset_ctr))
#                 logger.record_tabular("number_of_used", np.max(dataset_ctr))
#                 logger.record_tabular("episodes", num_episodes)
#                 logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
#                 logger.record_tabular("mean 100 episode length", mean_100ep_length)
#                 logger.record_tabular("lives", env.env.ale._used_lives)
#                 logger.dump_tabular()

#                 # Reset tracking stats
#                 policy_losses = []
#                 value_losses = []
#                 entropies = []
#                 kls = []
#                 dataset_ctr = np.zeros(shape=(len(env.env.ale.metadata)))
#                 dataset_ctr[dataset_idx] += 1

#             # Save checkpoint if performance improved
#             if (checkpoint_freq is not None and t > learning_starts and
#                     num_episodes > 100 and t % checkpoint_freq == 0):
#                 if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
#                     if print_freq is not None:
#                         logger.log("Saving model due to mean reward increase: {} -> {}".format(
#                             saved_mean_reward, mean_100ep_reward))
#                     save_state(model_file)
#                     model_saved = True
#                     saved_mean_reward = mean_100ep_reward
#                 if mean_100ep_reward > mean_100ep_reward_running:
#                     save_state(model_file + '-running')
#                     if print_freq is not None:
#                         logger.log("Saving model due to running mean reward increase: {} -> {}".format(
#                             mean_100ep_reward_running, mean_100ep_reward))

#         # Load best model at the end of training
#         if model_saved:
#             if print_freq is not None:
#                 logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
#             load_state(model_file)

#     # Save final model
#     save_state(model_file + '-final')
#     return act

# def predict_hpo_config(hpo_act_function, n_t_dim, max_seq_len):
#     """
#     Function to generate HPO config prediction using trained agent

#     Parameters
#     ----------
#     hpo_act_function: function
#         Trained HPO action function
#     n_t_dim: int
#         Number of hyperparameter dimensions
#     max_seq_len: int
#         Maximum sequence length for observation

#     Returns
#     -------
#     config: list
#         Predicted hyperparameter configuration
#     """
#     # Create a dummy observation filled with zeros
#     obs = np.zeros((max_seq_len, n_t_dim))

#     # Generate a configuration by selecting actions sequentially
#     config = []
#     for step in range(n_t_dim):
#         # Get action for the current step
#         action = hpo_act_function(obs, step)
#         config.append(action)

#         # Update observation for next step
#         if step < n_t_dim - 1:
#             obs[step + 1, step] = action

#     return config

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import common.tf_util as U
from agent.deepq.simple import ActWrapper
import os
import wandb
from common.tf_util import ortho_init
import tempfile
import logger
from agent.deepq.utils import ObservationInput
from common.tf_util import load_state, save_state
import copy

def preprocess_observation(obs):
    obs = np.array(obs, dtype=np.float32)
    orig_shape = obs.shape
    obs_flat   = obs.reshape(-1, obs.shape[-1])
    main_obs   = obs_flat[:, :4]
    meta       = obs_flat[:, 4:].copy()         # <- idempotent

    meta[np.isclose(meta, -999999)] = -1.0
    meta[np.isclose(meta,  999999)] =  1.0
    meta[np.isclose(meta, 1000)]    =  1.0      # <- correct min-max

    meta[:, 4] = 2 * meta[:, 4] / 500 - 1
    meta[:, 0] = 2 * (meta[:, 0]-1)/(99-1) - 1
    meta[:, 2] = 2 * (meta[:, 2]-3)/(99-3) - 1
    return np.concatenate([main_obs, meta], axis=1).reshape(orig_shape)

def scope_vars(scope, trainable_only=False):
    """Get variables inside a scope
    Parameters
    ----------
    scope: str or VariableScope
        scope in which the variables reside.
    trainable_only: bool
        whether or not to return only the variables that were marked as trainable.
    Returns
    -------
    vars: [tf.Variable]
        list of variables in `scope`.
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def lstm_to_mlp(cell, hiddens, max_length, aktiv=tf.nn.tanh, dueling=False):
    """
    Creates an MLP from an LSTM embedding for PPO - Adapted from DeepQ architecture
    
    Parameters
    ----------
    cell: tuple
        cell[0] is the number of LSTM units (nh)
        cell[1] is the number of input features (nin) 
        cell[2] is the number of metafeatures (nfeatures)
    hiddens: list
        List of hidden layer sizes
    max_length: int
        Maximum sequence length
    aktiv: tf activation function
        Activation function
    dueling: bool
        Whether to use dueling architecture (not used in PPO)

    Returns
    -------
    actor_network: function
        Actor network function
    value_network: function
        Value network function
    """
    nh, nin, nfeatures = cell

    def actor_network(x, seq, num_actions, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope("lstmnet"):
                # LSTM weights - same as DeepQ
                wx = tf.get_variable("wx", [nin, nh*4], initializer=ortho_init(1.0))
                wh = tf.get_variable("wh", [nh, nh*4], initializer=ortho_init(1.0))
                wh0 = tf.get_variable("wh0", [nfeatures, nh], initializer=ortho_init(1.0))
                b = tf.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))
                
                # Initialize LSTM state
                c = tf.zeros(shape=(tf.shape(x)[0], nh), dtype=tf.float32)
                h = tf.matmul(x[:, 0, -nfeatures:], wh0)  # Initialize with metafeatures
                
                # Process sequence
                out = []
                for idx in range(max_length):
                    # Extract input features for this timestep
                    x_t = x[:, idx, :nin]
                    
                    # LSTM computation - same as DeepQ
                    z = tf.matmul(tf.zeros(shape=tf.shape(x_t)) if idx == 0 else x_t, wx) + tf.matmul(h, wh) + b
                    i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
                    i = tf.nn.sigmoid(i)
                    f = tf.nn.sigmoid(f)
                    o = tf.nn.sigmoid(o)
                    u = tf.tanh(u)
                    c = f*c + i*u
                    h = o*tf.tanh(c)
                    out.append(h)
                
                # Stack and transpose outputs
                out = tf.transpose(tf.stack(out), perm=[1, 0, 2])
                
                # Extract output at current sequence position - same as DeepQ
                batch_size = tf.shape(out)[0]
                index = tf.range(0, batch_size) * max_length + (seq - 1)
                out = tf.gather(tf.reshape(out, [-1, nh]), index)
                lstm_out = tf.layers.dense(out, units=nh, activation=aktiv)
            
            # Action head
            with tf.variable_scope("action_value"):
                action_out = lstm_out
                for hidden in hiddens:
                    action_out = tf.layers.dense(action_out, units=hidden, activation=None)
                    action_out = aktiv(action_out)
                action_scores = tf.layers.dense(action_out, units=num_actions, activation=None)
            
            return action_scores

    def value_network(x, seq, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope("lstmnet"):  # Reuse LSTM weights
                # LSTM weights - same as DeepQ
                wx = tf.get_variable("wx", [nin, nh*4], initializer=ortho_init(1.0))
                wh = tf.get_variable("wh", [nh, nh*4], initializer=ortho_init(1.0))
                wh0 = tf.get_variable("wh0", [nfeatures, nh], initializer=ortho_init(1.0))
                b = tf.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))
                
                # Initialize LSTM state
                c = tf.zeros(shape=(tf.shape(x)[0], nh), dtype=tf.float32)
                h = tf.matmul(x[:, 0, -nfeatures:], wh0)  # Initialize with metafeatures
                
                # Process sequence
                out = []
                for idx in range(max_length):
                    # Extract input features for this timestep
                    x_t = x[:, idx, :nin]
                    
                    # LSTM computation - same as DeepQ
                    z = tf.matmul(tf.zeros(shape=tf.shape(x_t)) if idx == 0 else x_t, wx) + tf.matmul(h, wh) + b
                    i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
                    i = tf.nn.sigmoid(i)
                    f = tf.nn.sigmoid(f)
                    o = tf.nn.sigmoid(o)
                    u = tf.tanh(u)
                    c = f*c + i*u
                    h = o*tf.tanh(c)
                    out.append(h)
                
                # Stack and transpose outputs
                out = tf.transpose(tf.stack(out), perm=[1, 0, 2])
                
                # Extract output at current sequence position - same as DeepQ
                batch_size = tf.shape(out)[0]
                index = tf.range(0, batch_size) * max_length + (seq - 1)
                out = tf.gather(tf.reshape(out, [-1, nh]), index)
                lstm_out = tf.layers.dense(out, units=nh, activation=aktiv)
            
            # Value head
            with tf.variable_scope("state_value"):
                state_out = lstm_out
                for hidden in hiddens:
                    state_out = tf.layers.dense(state_out, units=hidden, activation=None)
                    state_out = tf.nn.relu(state_out)
                state_score = tf.layers.dense(state_out, units=1, activation=None)
            
            return state_score

    return actor_network, value_network


def build_policy(make_obs_ph, actor_network, value_network, num_actions, scope="ppo", entropy_coef=0.1, reuse=None):
    """
    Build PPO policy and value networks

    Parameters
    ----------
    make_obs_ph: function
        function that creates observation placeholder
    actor_network: function
        function that takes observation tensor and returns action logits
    value_network: function
        function that takes observation tensor and returns value estimate
    num_actions: int
        number of possible actions
    scope: str
        name of the scope
    reuse: bool
        whether or not to reuse existing scope

    Returns
    -------
    act: function
        function to select and return actions
    train: function
        function to train the PPO policy
    update_old_policy: function
        function to update target policy for stable updates
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Placeholders
        observations_ph = make_obs_ph("observation")
        seq_t_ph = tf.placeholder(tf.int32, [None], name="seq_t")
        actions_ph = tf.placeholder(tf.int32, [None], name="actions")
        advantages_ph = tf.placeholder(tf.float32, [None], name="advantages")
        returns_ph = tf.placeholder(tf.float32, [None], name="returns")
        old_neglogp_ph = tf.placeholder(tf.float32, [None], name="old_neglogp")
        old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred")
        lr_ph = tf.placeholder(tf.float32, [], name="learning_rate")
        clip_param_ph = tf.placeholder(tf.float32, [], name="clip_param")
        entropy_coef_ph = tf.placeholder(tf.float32, [], name="entropy_coef")

        # Policy network (Actor)
        with tf.variable_scope("policy"):
            policy_logits = actor_network(observations_ph.get(), seq_t_ph, num_actions, scope="policy_net")

            # Single, stable policy distribution
            policy_logits = tf.clip_by_value(policy_logits, -100.0, 100.0)  # Prevent extreme values
            policy_dist = tf.distributions.Categorical(logits=policy_logits)
            action = policy_dist.sample()
            neglogp = -policy_dist.log_prob(action)
            neglogp_actions = -policy_dist.log_prob(actions_ph)

        # Value network (Critic)
        with tf.variable_scope("value"):
            value_pred = value_network(observations_ph.get(), seq_t_ph, scope="value_net")
            value_pred = tf.squeeze(value_pred, axis=1)

        # Old (target) policy, used for KL and ratio calculation
        with tf.variable_scope("old_policy"):
            old_policy_logits = actor_network(observations_ph.get(), seq_t_ph, num_actions, scope="old_policy_net")
            old_policy_probs = tf.nn.softmax(old_policy_logits)
            old_policy_dist = tf.distributions.Categorical(probs=old_policy_probs)
            old_neglogp = -old_policy_dist.log_prob(actions_ph)

        # Get trainable variables for all networks
        def _sorted(sub):
            return sorted(scope_vars(scope + "/" + sub, trainable_only=True), key=lambda v: v.name)
        policy_vars = _sorted("policy")
        value_vars  = _sorted("value")
        old_policy_vars = _sorted("old_policy")

        # print("Value network variables:")
        # for var in value_vars:
        #     print(var.name)

        # Loss calculations
        # Policy loss with clipping
        ratio = tf.exp(
            old_neglogp_ph - neglogp_actions)  # ratio between old and new policy, should be one at the first iteration
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_param_ph, 1.0 + clip_param_ph)
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages_ph, clipped_ratio * advantages_ph))

        # Value loss
        value_clipped = old_vpred_ph + tf.clip_by_value(value_pred - old_vpred_ph, -clip_param_ph, clip_param_ph)
        value_loss1 = tf.square(value_pred - returns_ph)
        value_loss2 = tf.square(value_clipped - returns_ph)
        # Standard PPO value loss
        value_loss = 0.5 * tf.reduce_mean(tf.square(value_pred - returns_ph))

        # Entropy loss to encourage exploration
        entropy = tf.reduce_mean(policy_dist.entropy())

        # Total loss
        loss = policy_loss + 0.5 * value_loss - entropy_coef_ph * entropy

        # KL divergence for adaptive learning rate
        approxkl = tf.reduce_mean(old_neglogp_ph - neglogp_actions)

        # Create optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_ph)

        # Gradient clipping
        grads_and_vars = optimizer.compute_gradients(loss, var_list=policy_vars + value_vars)
        grads, vars = zip(*grads_and_vars)
        # print("Variables in optimizer:")
        # for grad, var in grads_and_vars:
        #     print(var.name)
        grads_clipped, _ = tf.clip_by_global_norm(grads, 1)
        optimize_expr = optimizer.apply_gradients(list(zip(grads_clipped, vars)))
        # For debugging: get the norm of each gradient
        grad_norms = [tf.norm(g) if g is not None else tf.constant(0.0) for g in grads_clipped]


        # Update old policy
        update_old_policy_expr = []
        for var, var_old in zip(sorted(policy_vars, key=lambda v: v.name),
                                sorted(old_policy_vars, key=lambda v: v.name)):
            update_old_policy_expr.append(var_old.assign(var))
        update_old_policy_expr = tf.group(*update_old_policy_expr)

        # Create callable functions
        def act(obs, seq):
            """Take an observation and return an action and its value"""
            a, v, nlp = sess.run([action, value_pred, neglogp],
                                 feed_dict={observations_ph.get(): obs, seq_t_ph: seq})
            return a, v, nlp

        def compute_values(obs, seq):
            """Compute value estimates for a batch of observations"""
            return sess.run(value_pred,
                            feed_dict={observations_ph.get(): obs, seq_t_ph: seq})

        def train(lr, clip_param, obs, seq, actions, values, returns, neglogpacs, advantages, entropy_coef):
            """Update policy and value networks"""
            td_map = {
                lr_ph: lr,
                clip_param_ph: clip_param,
                observations_ph.get(): obs,
                seq_t_ph: seq,
                actions_ph: actions,
                advantages_ph: advantages,
                returns_ph: returns,
                old_neglogp_ph: neglogpacs,
                old_vpred_ph: values,
                entropy_coef_ph: entropy_coef,
            }

            policy_loss_val, value_loss_val, entropy_val, approxkl_val, grad_norms_val, _ = sess.run(
                [policy_loss, value_loss, entropy, approxkl, grad_norms, optimize_expr],
                feed_dict=td_map
            )

            # print("Value loss:", value_loss_val)
            # for grad, var in zip(grad_norms_val, vars):
            #     if grad is not None:
            #         print(var.name, "grad norm:", np.linalg.norm(grad))
            #     else:
            #         print(var.name, "grad is None")

            return policy_loss_val, value_loss_val, entropy_val, approxkl_val

        def update_old_policy():
            """Update target policy to current policy for stable updates"""
            return sess.run(update_old_policy_expr)

        # Save session reference
        sess = tf.get_default_session()

    return act, train, update_old_policy, compute_values


class PPOBuffer:
    """Buffer for storing trajectory data for PPO training with flexible capacity"""

    def __init__(self, size, obs_shape, gamma=0.99, lam=0.95):
        self.obs = np.zeros((size,) + obs_shape, dtype=np.float32)
        self.seq = np.zeros(size, dtype=np.int32)
        self.actions = np.zeros(size, dtype=np.int32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.neglogpacs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)

        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx = 0, 0
        self.max_size = size
        self._next_idx = 0
        self._full = False

    def store(self, obs, seq, action, reward, value, neglogpac, done):
        """Store a single transition"""
        assert self.ptr < self.max_size
        self.obs[self.ptr] = obs
        self.seq[self.ptr] = seq
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = np.clip(reward, -10., 10.)
        self.values[self.ptr] = value
        self.neglogpacs[self.ptr] = neglogpac
        self.dones[self.ptr] = done
        self.ptr += 1

        # Update _next_idx and check if buffer is full
        self._next_idx = self.ptr % self.max_size
        if self.ptr >= self.max_size:
            self._full = True

    def finish_path(self, last_val=0):
        """
        Compute advantages and returns for a completed trajectory
        using GAE (Generalized Advantage Estimation)
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        path_len = self.ptr - self.path_start_idx

        # Handle empty paths
        if path_len <= 0:
            self.path_start_idx = self.ptr
            return

        rewards = np.append(self.rewards[path_slice], last_val)
        values = np.append(self.values[path_slice], last_val)
        dones = np.append(self.dones[path_slice], 0)

        # Safe GAE calculation - prevent NaN propagation
        deltas = rewards[:-1] + self.gamma * values[1:] * (1 - dones[:-1]) - values[:-1]

        # Clip extreme values
        deltas = np.clip(deltas, -100.0, 100.0)

        # Calculate advantages with safety
        gae = 0
        advantages = np.zeros(path_len)

        for t in reversed(range(path_len)):
            gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
            # Prevent extreme values
            gae = np.clip(gae, -100.0, 100.0)
            advantages[t] = gae

        # Store advantages and compute returns
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = advantages + self.values[path_slice]

        # Clip returns to prevent extreme values
        self.returns[path_slice] = np.clip(self.returns[path_slice], -10.0, 10.0)

        self.path_start_idx = self.ptr

    def get(self):
        """Get stored data for training, using available data"""
        # Use available data even if buffer is not completely full
        available_size = self.ptr if not self._full else self.max_size

        if available_size < 1:
            raise ValueError("No data available in buffer")

        # If buffer wrapped around, use data from 0 to _next_idx
        if self._full:
            indices = np.arange(self.ptr) if not self._full else \
            np.concatenate([np.arange(self._next_idx, self.max_size),
                          np.arange(0, self._next_idx)])
        else:
            indices = np.arange(0, self.ptr)

        # Normalize advantages
        if np.any(np.isnan(self.advantages[indices])):
            self.advantages[indices] = np.nan_to_num(self.advantages[indices])

            # Safe advantage normalization
        if len(indices) > 1:  # Only normalize if we have multiple samples
            adv_mean = np.mean(self.advantages[indices])
            adv_std = np.std(self.advantages[indices])

            # Prevent division by zero
            if adv_std < 1e-8:
                adv_std = 1.0

            self.advantages[indices] = (self.advantages[indices] - adv_mean) / adv_std

            # Final safety check for NaNs
        for array in [self.obs[indices], self.seq[indices], self.actions[indices],
                      self.values[indices], self.returns[indices],
                      self.neglogpacs[indices], self.advantages[indices]]:
            if np.any(np.isnan(array)):
                # Replace NaNs with safe values
                np.nan_to_num(array, copy=False)

        # Prepare data for training
        return (
            self.obs[indices],
            self.seq[indices],
            self.actions[indices],
            self.values[indices],
            self.returns[indices],
            self.neglogpacs[indices],
            self.advantages[indices]
        )

    def clear(self):
        """Reset buffer"""
        self.ptr, self.path_start_idx = 0, 0
        self._full = False

    def __len__(self):
        """Return current buffer size"""
        return self.ptr if not self._full else self.max_size


def learn(env,
          actor_network,
          value_network,
          max_lives=1,
          lr=3e-4,
          max_timesteps=int(1e6),
          buffer_size=2048,
          train_freq=1,
          batch_size=64,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=0.99,
          lam=0.95,
          clip_ratio=0.2,
          train_epochs=10,
          N_t=4,
          callback=None,
          entropy_coef=0.1):
    """
    Train a PPO agent on the HPO environment

    Parameters
    ----------
    env: gym.Env
        Environment to train on
    actor_network: function
        Network that takes observations and returns action logits
    value_network: function
        Network that takes observations and returns value estimates
    max_lives: int
        Maximum number of lives before switching dataset
    lr: float
        Learning rate
    max_timesteps: int
        Total number of timesteps to train for
    buffer_size: int
        Size of the replay buffer
    train_freq: int
        How often to run PPO updates (in steps)
    batch_size: int
        Size of minibatches for training
    print_freq: int
        How often to print progress information
    checkpoint_freq: int
        How often to save model checkpoints (timesteps)
    checkpoint_path: str
        Path to save model checkpoints
    learning_starts: int
        How many steps to collect before starting training
    gamma: float
        Discount factor
    lam: float
        GAE lambda parameter
    clip_ratio: float
        PPO clipping parameter
    train_epochs: int
        Number of PPO epochs per update
    N_t: int
        Number of hyperparameter selection steps
    callback: function
        Optional callback function to monitor training

    Returns
    -------
    act: ActWrapper
        Wrapper over act function
    """
    # Create all the functions necessary to train the model
    sess = tf.Session()
    sess.__enter__()

    # Create observation placeholder
    def make_obs_ph(name):
        return ObservationInput(env.observation_space, name=name)

    # Get observation space shape and action space size
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    # Build policy networks and training functions
    act, train, update_old_policy, compute_values = build_policy(
        make_obs_ph=make_obs_ph,
        actor_network=actor_network,
        value_network=value_network,
        num_actions=num_actions,
        entropy_coef=entropy_coef
        # optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        # session=sess
    )

    # Store act parameters for saving/loading
    act_params = {
        'make_obs_ph': make_obs_ph,
        'actor_network': actor_network,
        'value_network': value_network,
        'num_actions': num_actions,
        'N_t': N_t
    }

    # Create ActWrapper that can be saved/loaded
    act = ActWrapper(act, act_params)

    # Initialize variables
    U.initialize()
    update_old_policy()  # Sync old policy with new policy

    # Create buffer for storing experience
    buffer = PPOBuffer(buffer_size, obs_shape, gamma, lam)

    # Initialize tracking variables
    episode_rewards = [0.0]
    episode_ei = [0.0]  # Including ei tracking for consistency
    episode_lengths = [1]
    policy_losses = []
    value_losses = []
    entropies = []
    kls = []
    saved_mean_reward = None

    # Initialize dataset tracking as in DeepQ implementation
    dataset_idx = 0
    dataset_ctr = np.zeros(shape=(len(env.env.ale.metadata)))
    dataset_ctr[dataset_idx] += 1

    # create a lambda that falls to zero by max_timesteps
    lr_schedule = lambda step: lr * (1. - min(1., step / max_timesteps))
    # entropy coefficient: start at entropy_coef, decay to 0
    entropy_schedule = lambda step: entropy_coef * max(0.01, 1. - step / (2 * max_timesteps))

    # Initialize environment state using the same pattern as DeepQ
    state = env.reset()
    obs = np.zeros(shape=env.observation_space.shape)
    obs[0, :] = np.append(np.repeat(np.NaN, repeats=N_t), env.env.ale.metadata[dataset_idx]['features']).reshape(1, -1)
    obs = np.nan_to_num(obs, copy=True, nan=0.0)  # Replace NaN with 0
    reset = True
    prev_r = 0

    wandb.init(project="RL-HPO-Visualization",
               name=f"PPO-{env.env_id if hasattr(env, 'env_id') else 'unknown'}",
               config={
                   "algorithm": "PPO",
                   "env_id": env.env_id if hasattr(env, 'env_id') else 'unknown',
                   "learning_rate": lr,
                   "buffer_size": buffer_size,
                   "batch_size": batch_size,
                   "gamma": gamma,
                   "lam": lam,
                   "clip_ratio": clip_ratio,
                   "train_epochs": train_epochs
               })

    wandb.log({"policy_loss_mean_100": 0,
               "value_loss_mean_100": 0,
               "entropy_mean_100": 0,
               "approx_kl_mean_100": 0,
               "episode_reward": 0,
               "episode_length": 1,
               "mean_100ep_reward": 0,
               "episode_length_mean_100": 0,
               "timesteps": 0
               })

    def wandb_callback(locals, globals):
        # Log metrics at the end of each episode
        if len(locals['episode_rewards']) > 0:  # Log every 100 timesteps

            last_100_rewards = locals['episode_rewards'][-100:]
            last_100_lengths = locals['episode_lengths'][-100:]
            last_100_policy_losses = locals.get('policy_losses', [0])[-100:]
            last_100_value_losses = locals.get('value_losses', [0])[-100:]
            last_100_entropies = locals.get('entropies', [0])[-100:]
            last_100_kls = locals.get('kls', [0])[-100:]

            wandb.log({
                "episode_reward": locals['episode_rewards'][-1],
                "episode_length": locals['episode_lengths'][-1],
                "mean_100ep_reward": np.mean(last_100_rewards),
                "episode_length_mean_100": np.mean(last_100_lengths),
                "policy_loss_mean_100": np.mean(last_100_policy_losses),
                "value_loss_mean_100": np.mean(last_100_value_losses),
                "entropy_mean_100": np.mean(last_100_entropies),
                "approx_kl_mean_100": np.mean(last_100_kls),
                "timesteps": locals['t']
            })
        return False

    # Combine the original callback with wandb callback
    def combined_callback(locals, globals):
        if callback is not None:
            callback_result = callback(locals, globals)
            if callback_result:
                return True
        return wandb_callback(locals, globals)

    # Save/load model handling
    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td
        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_state(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True

        # Main training loop
        for t in range(max_timesteps):

            # Get current sequence position for observation - match DeepQ's pattern
            seq = env.env._get_ep_len()

            # Get action from policy - match DeepQ's pattern of passing sequence
            action, value, neglogp = act(np.array(obs)[None], [seq])
            action = action[0]  # Extract scalar from array
            value = value[0]
            neglogp = neglogp[0]

            # Take step in environment using DeepQ's pattern
            new_state, rew, done, _ = env.step(action.item(), dataset_id=dataset_idx)

            # Update observation using DeepQ's pattern
            new_obs = copy.copy(obs)
            new_obs[env.env._get_ep_len(), :] = np.append(new_state, np.append(rew, env.env.ale.metadata[dataset_idx][
                'features'])).reshape(1, -1)

            # Store in PPO buffer
            buffer.store(obs, seq, action, rew, value, neglogp, float(done))

            # Update tracking variables
            obs = new_obs
            episode_rewards[-1] += rew
            prev_r = copy.copy(rew)

            # Handle episode termination - match DeepQ's reset pattern
            if done:
                episode_lengths[-1] = env.env._get_ep_len()

                # Finish path for PPO calculations
                buffer.finish_path()

                # Reset state using DeepQ's pattern
                state = env.reset()
                obs = np.zeros(shape=env.observation_space.shape)
                obs[0, :] = np.append(np.repeat(np.NaN, repeats=N_t),
                                      env.env.ale.metadata[dataset_idx]['features']).reshape(1, -1)
                obs = np.nan_to_num(obs, copy=True, nan=0.0)

                # Update dataset tracking as in DeepQ
                env.env.ale._used_lives += 1
                if env.env.ale._used_lives % max_lives == 0:
                    dataset_idx += 1
                    dataset_idx = dataset_idx % len(env.env.ale.metadata)
                    dataset_ctr[dataset_idx] += 1

                # Start new episode
                episode_lengths.append(1)
                episode_rewards.append(0.0)
                episode_ei.append(0.0)
                reset = True
                prev_r = 0

            # Check if we need to run PPO updates (similar to DeepQ training check)
            if t > learning_starts and t % train_freq == 0 and len(buffer) >= batch_size:
                # Get data from buffer
                obs_buffer, seq_buffer, act_buffer, val_buffer, ret_buffer, old_neglogp_buffer, adv_buffer = buffer.get()

                # Tracking stats
                avg_policy_loss = 0
                avg_value_loss = 0
                sum_entropy = 0.0
                n_updates = 0
                avg_approxkl = 0

                policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ppo/value")
                weights_before = tf.get_default_session().run(policy_vars)

                # Run PPO updates for multiple epochs
                adaptive_lr = lr * max(0.01, 1. - t / max_timesteps)
                for _ in range(train_epochs):
                    # Shuffle data
                    buffer_size = len(buffer)
                    indices = np.arange(buffer_size)
                    np.random.shuffle(indices)

                    # Train in mini-batches
                    for start in range(0, buffer_size, batch_size):
                        end = start + batch_size
                        batch_indices = indices[start:end]
                        if len(batch_indices) < batch_size:  # Skip small last batch
                            continue

                        # print("Value network input batch stats:",
                        #       "mean", np.mean(obs_buffer[batch_indices]),
                        #       "std", np.std(obs_buffer[batch_indices]),
                        #       "min", np.min(obs_buffer[batch_indices]),
                        #       "max", np.max(obs_buffer[batch_indices]))

                        # Update policy and value networks
                        policy_loss, value_loss, entropy, approxkl = train(
                            lr=adaptive_lr,
                            clip_param=clip_ratio,
                            obs=obs_buffer[batch_indices],
                            seq=seq_buffer[batch_indices],
                            actions=act_buffer[batch_indices],
                            advantages=adv_buffer[batch_indices],
                            returns=ret_buffer[batch_indices],
                            neglogpacs=old_neglogp_buffer[batch_indices],
                            values=val_buffer[batch_indices],
                            entropy_coef=entropy_schedule(t)
                        )

                        if approxkl > 0.01:
                            adaptive_lr *= 0.8

                        # Track stats
                        avg_policy_loss += policy_loss / train_epochs
                        avg_value_loss += value_loss / train_epochs
                        sum_entropy += entropy
                        n_updates += 1
                        avg_approxkl += approxkl / train_epochs

                # Update old policy after all updates
                update_old_policy()

                weights_after = tf.get_default_session().run(policy_vars)
                # for w_b, w_a in zip(weights_before, weights_after):
                #     print("Policy weight diff:", np.sum(np.abs(w_b - w_a)))

                # Store stats for logging
                policy_losses.append(avg_policy_loss)
                value_losses.append(avg_value_loss)
                mean_entropy = sum_entropy / max(1, n_updates)
                entropies.append(mean_entropy)
                kls.append(avg_approxkl)

                if combined_callback is not None:
                    if combined_callback(locals(), globals()):
                        break

                # Clear buffer after updates
                buffer.clear()

            # Print progress info - match DeepQ's logging pattern
            mean_100ep_length = round(
                np.mean(episode_lengths[-101:-1]) if len(episode_lengths) > 101 else np.mean(episode_lengths), 4)
            mean_100ep_reward = round(
                np.mean(episode_rewards[-101:-1]) if len(episode_rewards) > 101 else np.mean(episode_rewards), 4)
            mean_100ep_reward_running = round(
                np.mean(episode_rewards[-201:-100]) if len(episode_rewards) > 201 else np.mean(episode_rewards), 4)
            num_episodes = len(episode_rewards)

            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("policy_loss", round(np.mean(policy_losses), 4) if policy_losses else 0)
                logger.record_tabular("value_loss", round(np.mean(value_losses), 4) if value_losses else 0)
                logger.record_tabular("entropy", round(np.mean(entropies), 4) if entropies else 0)
                logger.record_tabular("approx_kl", round(np.mean(kls), 4) if kls else 0)
                logger.record_tabular("steps", t)
                logger.record_tabular("most_used_dataset", np.argmax(dataset_ctr))
                logger.record_tabular("number_of_used", np.max(dataset_ctr))
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("mean 100 episode length", mean_100ep_length)
                logger.record_tabular("lives", env.env.ale._used_lives)
                logger.dump_tabular()

                # Reset tracking stats
                policy_losses = []
                value_losses = []
                entropies = []
                kls = []
                dataset_ctr = np.zeros(shape=(len(env.env.ale.metadata)))
                dataset_ctr[dataset_idx] += 1

            # Save checkpoint if performance improved
            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                            saved_mean_reward, mean_100ep_reward))
                    save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
                if mean_100ep_reward > mean_100ep_reward_running:
                    save_state(model_file + '-running')
                    if print_freq is not None:
                        logger.log("Saving model due to running mean reward increase: {} -> {}".format(
                            mean_100ep_reward_running, mean_100ep_reward))

        # Load best model at the end of training
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_state(model_file)

    # Save final model
    save_state(model_file + '-final')
    return act

def predict_hpo_config(hpo_act_function, n_t_dim, max_seq_len):
    """
    Function to generate HPO config prediction using trained agent

    Parameters
    ----------
    hpo_act_function: function
        Trained HPO action function
    n_t_dim: int
        Number of hyperparameter dimensions
    max_seq_len: int
        Maximum sequence length for observation

    Returns
    -------
    config: list
        Predicted hyperparameter configuration
    """
    # Create a dummy observation filled with zeros
    obs = np.zeros((max_seq_len, n_t_dim))

    # Generate a configuration by selecting actions sequentially
    config = []
    for step in range(n_t_dim):
        # Get action for the current step
        action = hpo_act_function(obs, step)
        config.append(action)

        # Update observation for next step
        if step < n_t_dim - 1:
            obs[step + 1, step] = action

    return config