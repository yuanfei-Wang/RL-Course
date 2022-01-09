from gym.envs.registration import register
register(
     id='Env2048-v0',
     entry_point='gym2048.Env2048:Env2048',
     max_episode_steps=5000,
)
register(
     id='Env2048soft-v0',
     entry_point='gym2048.Env2048:Env2048soft',
     max_episode_steps=5000,
)
register(
     id='Env2048onehot-v0',
     entry_point='gym2048.Env2048:Env2048onehot',
     max_episode_steps=5000,
)