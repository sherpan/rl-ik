import gym
import ik_2D_3DOF_arm
from stable_baselines import PPO1

filename = "ppo_ik_agent"
env = gym.make("ik-2D-3DOF-arm-v0")
model = PPO1.load(filename, verbose=0)
model.set_env(env)
success_episodes = 0
print("Starting Inference")

test = 1000
total_steps = 0
for x in range(0, test):
    is_done = False
    obs = env.reset()
    steps = 0
    while not is_done:
        action, _ = model.predict(obs)
        obs, reward, is_done,_ = env.step(action)
        steps = steps + 1
    if steps < 101:
        success_episodes = success_episodes + 1
        total_steps = total_steps + steps

print("Accuracy: " + str(100*(success_episodes/test)) +"%")
print("Avg Steps Per Successful Episode: " + str((total_steps/success_episodes)))
