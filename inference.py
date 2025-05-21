import numpy as np
import os
import gymnasium as gym
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # Keep this for TF1 compatibility
import argparse
import logger

# Assuming your 'agent' and 'common' directories are accessible
from agent import deepq
from common.misc_util import set_global_seeds
from common.cmd_util import make_meta_env
from common.utils import model_dir

# -------- Helper Functions for Prediction (Modified) --------

# of shape (1, expected_num_features) with appropriate normalization/scaling.
def prepare_initial_observation(max_seq_len: int, n_t_dim: int) -> np.ndarray:
    """
    Prepares the initial observation matrix (empty history).

    Args:
        max_seq_len: The maximum sequence length the model expects.
        n_t_dim: The dimension of the hyperparameter vectors (lambda).

    Returns:
        A numpy array representing the initial state, shape (max_seq_len, n_t_dim).
    """
    # Create an observation matrix representing an empty history.
    # Using NaNs is often safer if the model might treat zero as a valid value.
    initial_obs = np.full((max_seq_len, n_t_dim), np.nan, dtype=np.float32)
    # Note: If meta-features were concatenated into this obs matrix during
    # training, replicate that structure here. Assuming base obs is just sequence.
    return initial_obs

def predict_hpo_config(
    env_name: str,
    hpo_act_function, # The 'act' function from the trained agent
    n_t_dim: int,
    max_seq_len: int,
    ):
    """
    Predicts the best initial HPO configuration for a given Gymnasium environment,
    using the already trained agent's act function.

    Args:
        env_name: The name of the Gymnasium environment (e.g., "CartPole-v1").
        hpo_act_function: The loaded/trained act function.
        lambda_mapping: Numpy array mapping action index to lambda vector.
        n_f_dim: Dimension of meta-features.
        n_t_dim: Dimension of hyperparameter vectors (lambda).
        max_seq_len: Max sequence length expected by the model.
        param_names: Optional list of hyperparameter names for output dict.
    """
    print(f"\n--- Predicting HPO config for environment: {env_name} ---")
    target_env = gym.make("Taxi-v3")

    # NOTE: Ensure this function produces features compatible with training!
    meta_features = np.array([1,0,3,0,0,1,0,-999999,999999,0,0,7,2])

    # Prepare initial observation state (empty history)
    initial_obs = prepare_initial_observation(max_seq_len, n_t_dim)

    # Add batch dimension as expected by the agent's act function
    # Shape: (1, max_seq_len, N_t)
    obs_batch = initial_obs[None, :, :]

    # Sequence length for the initial state (empty history)
    initial_seq_len = [1] # Batch of sequence lengths

    # Get prediction using the trained agent's act function
    # Run in deterministic mode (stochastic=False, update_eps=0)
    predicted_action_index = hpo_act_function(
        obs_batch,
        initial_seq_len,
        stochastic=False,
        update_eps=0 # Use 0 epsilon for deterministic prediction
        )
    print(f"Predicted action index: {predicted_action_index}")
    return predicted_action_index

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='HPO Training environment ID', default='nnMeta-v40')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=0)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    parser.add_argument('--dueling', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e3)) # Training duration
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--learning_starts', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--target_network_update_freq', type=int, default=10)
    parser.add_argument('--cell', type=int, default=32) # LSTM cell size
    parser.add_argument('--nhidden', type=int, default=128) # MLP hidden size
    parser.add_argument('--ei', default='True') # Use Expected Improvement reward shaping?
    # Add argument for the new environment to predict for
    parser.add_argument('--predict_env', help='Gymnasium environment ID to predict HPO config for', default='CartPole-v1')
    # Add argument for Lambda mapping path (if needed)
    parser.add_argument('--lambda_path', help='Path to Lambda mapping .npy file', default='lambda_mapping.npy') # Example path


    args = parser.parse_args()
    args.buffer_size = args.learning_starts # Set buffer size based on learning starts

    # --- Determine N_t and N_f based on training env_id ---
    # These need to match the data the training env ('nnMetaEnv') uses
    if args.env_id.startswith('nn'):
        N_t = 4  # Hyperparameter vector dimension for 'nn' tasks
        N_f = 13 # Meta-feature dimension for 'nn' tasks
        print(f"Using N_t={N_t}, N_f={N_f} for env_id '{args.env_id}'")
    elif args.env_id.startswith('svm'):
        N_t = 7  # Hyperparameter vector dimension for 'svm' tasks
        N_f = 3  # Meta-feature dimension for 'svm' tasks
        print(f"Using N_t={N_t}, N_f={N_f} for env_id '{args.env_id}'")
    else:
        # Default or raise error if env_id prefix is unknown
        raise ValueError(f"Unknown env_id prefix: {args.env_id}. Cannot determine N_t and N_f.")

    checkpoint_path = model_dir(args)
    logger.configure(checkpoint_path) # Configure logger *before* TF graph potentially prints things
    set_global_seeds(args.seed)

    # --- Setup Training Environment ---
    # This is the nnMetaEnv used for training the HPO agent
    train_env = make_meta_env(args.env_id, seed=args.seed)
    # Extract max_sequence_length from the training env's observation space
    MAX_SEQUENCE_LENGTH = train_env.observation_space.shape[0]
    print(f"Training Environment Observation Space Shape: {train_env.observation_space.shape}")
    print(f"Using Max Sequence Length: {MAX_SEQUENCE_LENGTH}")


    # --- Define Q-Function Model ---
    model = deepq.models.lstm_to_mlp(
        cell=(args.cell, N_t, N_f), # Pass dimensions to the model builder
        aktiv=tf.nn.tanh,
        hiddens=[args.nhidden],
        max_length=MAX_SEQUENCE_LENGTH, # Use extracted max length
        dueling=bool(args.dueling),
    )

    # --- Train the HPO Agent ---
    print("\n--- Starting HPO Agent Training ---")
    # deepq.learn returns an ActWrapper instance containing the act function and params
    trained_act_wrapper = deepq.learn(
        train_env,
        q_func=model,
        lr=args.lr,
        max_timesteps=args.num_timesteps,
        buffer_size=args.buffer_size,
        exploration_fraction=0.1, # Example value
        exploration_final_eps=0.01, # Example value
        train_freq=args.train_freq,
        learning_starts=args.learning_starts,
        target_network_update_freq=args.target_network_update_freq,
        gamma=0.99, # Discount factor for HPO rewards
        prioritized_replay=bool(args.prioritized),
        prioritized_replay_alpha=args.prioritized_replay_alpha,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_path=checkpoint_path.replace('/logs/','/checkpoints/'), # Standard practice
        ei=eval(args.ei),
        N_t=N_t, # Pass N_t to learn function (as it seems to expect it)
        max_lives=1 # Adjust if your meta env uses lives differently
    )
    print("--- HPO Agent Training Finished ---")

    # --- Prepare for Prediction ---
    print("\n--- Preparing for Prediction ---")
    # Extract the raw act function from the wrapper returned by learn
    # This function operates within the existing TF session
    hpo_act_function = trained_act_wrapper._act

    # --- Perform Prediction ---
    predicted_config = predict_hpo_config(
        env_name=args.predict_env,
        hpo_act_function=hpo_act_function,
        n_t_dim=N_t,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        )

    print(predicted_config)

    # --- Cleanup ---
    train_env.close()
    # The TensorFlow session used by learn() will be implicitly closed when the script ends.
    print("Script finished.")


if __name__ == '__main__':
    main()