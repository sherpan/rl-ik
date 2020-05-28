import gym
import ik_2D_3DOF_arm
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1

total_timesteps = 3000000
filename = "ppo_ik_agent"
env = gym.make("ik-2D-3DOF-arm-v0")
model = PPO1(MlpPolicy, env,  verbose=1,  tensorboard_log="./tensorboard/")
print("Starting to Train")
model.learn(total_timesteps=total_timesteps)
print("Done Training, Saving model")
model.save(filename)
