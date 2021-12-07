import gym2048
import gym

env = gym.make("Env2048-v0")
print(env.action_space.shape, env.observation_space.shape)
state = env.reset()
print(state)
print(env.observation_space.contains(state))
print(env.render())
env.step(3)
print(env.game.actions[3])
print(env.render())
env.step(0)
print(env.game.actions[0])
print(env.render())