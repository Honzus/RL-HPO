from agent.ppo.build_graph import lstm_to_mlp, PPOBuffer, learn, build_policy

def wrap_atari_ppo(env):
    from common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)