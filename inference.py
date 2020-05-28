import gym
import ik_2D_3DOF_arm
from stable_baselines import PPO1

filename = "ppo_ik_agent"
env = gym.make("ik-2D-3DOF-arm-v0")
model = PPO1.load(filename, verbose=0)
model.set_env(env)

print("Starting Inference")
for x in range(0, 3):
    is_done = False
    obs = env.reset()
    env.render()
    steps = 0
    while not is_done:
        action, _ = model.predict(obs)
        obs, reward, is_done,_ = env.step(action)

        env.render()
        steps = steps + 1
    print(steps)
    if steps < 101:
        print("Total Steps taken to reach goal: ", steps)
    else:
        print("Goal Not Reached")
