import gym
import ik_2D_3DOF_arm

env = gym.make("ik-2D-3DOF-arm-v0")

is_done = False
env.render()
while not is_done:
    action = env.action_space.sample()
    obs, reward, is_done,_ = env.step(action)
    print(obs)
    print(reward)
    env.render()
