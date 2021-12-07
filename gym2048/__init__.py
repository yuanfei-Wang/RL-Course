from gym.envs.registration import register
register(
     id='Env2048-v0',
     entry_point='gym2048.Env2048:Env2048',
     max_episode_steps=5000,
)