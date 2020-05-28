import gym
from gym import spaces
import numpy as np
import math
import matplotlib.pyplot as plt
import random


class IK2D3DOFARMEnv(gym.Env):

    _curr_pose = np.zeros(2)
    _goal_pose = np.zeros(2)
    _a1 = 2
    _a2 = 1
    _a3 = 0.5
    _q = np.zeros(3)
    _EPISODE_LIMIT = 100
    _episode_no = 0


    def __init__(self):

        self.action_space =  spaces.MultiDiscrete([ 11, 11, 11 ])

        self._reach = self._a1 + self._a2 + self._a3

        low_state = np.ones(4) * -1* self._reach
        high_state = np.ones(4) * 1* self._reach
        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)

        self._horizon = 0

        goal_q = np.array([random.randint(-170, 170), random.randint(-170, 170), random.randint(-170, 170)])
        self._goal_pose, _, _ = self._FK(goal_q)

        self._q = [90, 0, 0]



    def step(self, action):

        self._horizon = self._horizon + 1
        q_delta = action - 5

        if(self._check_limits(q_delta)):
            self._q = self._q + q_delta
            ee, _, _ = self._FK(self._q)
            reward = self._dist()
        else:
            reward = -2 * self._reach
            ee, _, _ = self._FK(self._q)

        obs = np.concatenate((self._goal_pose, ee))


        if reward > - 0.08 :
            done = True
            reward = 1000

        elif self._horizon > self._EPISODE_LIMIT:
            done = True
            reward = - 10

        else:
            done = False

        info = {}

        return obs, reward, done, info


    def reset(self):
        self._horizon = 0
        goal_q = np.array([random.randint(-170, 170), random.randint(-170, 170), random.randint(-170, 170)])
        #goal_q = np.array([45, 45, 45])
        self._goal_pose, _, _ = self._FK(goal_q)
        self._q = [90, 0, 0]
        ee, _, _ = self._FK(self._q)
        obs = np.concatenate((self._goal_pose, ee))
        self._episode_no  = self._episode_no + 1

        return obs


    def _FK(self, q):
        j2 = np.zeros(2)
        j3 = np.zeros(2)
        ee = np.zeros(2)
        j2[0] = self._a1*math.cos(math.radians(q[0]))
        j2[1] = self._a1*math.sin(math.radians(q[0]))
        j3[0] = self._a2*math.cos(math.radians(q[1] + q[0])) + j2[0]
        j3[1] = self._a2*math.sin(math.radians(q[1] + q[0])) + j2[1]
        ee[0] = self._a3*math.cos(math.radians(q[2] + q[1] + q[0])) + j3[0]
        ee[1] = self._a3*math.sin(math.radians(q[2] + q[1] + q[0])) + j3[1]
        return ee, j3, j2

    def _check_limits(self, q_delta):
        temp_q = q_delta + self._q
        return np.logical_and(temp_q > -180, temp_q < 180).all()

    def _dist(self):
        ee, _, _ = self._FK(self._q)
        return -1 * math.sqrt(math.pow(self._goal_pose[0] - ee[0], 2) +
                math.pow(self._goal_pose[1] - ee[1], 2) * 1.0)

    def render(self):
        plt.ylim(-self._reach - 0.5, self._reach + 0.5)
        plt.xlim(-self._reach - 0.5, self._reach + 0.5)
        plt.plot(self._goal_pose[0], self._goal_pose[1], 'ro')
        ee, j3, j2 = self._FK(self._q)
        plt.plot([0, j2[0]], [0, j2[1]])
        plt.plot([j2[0], j3[0]], [j2[1], j3[1]])
        plt.plot([j3[0], ee[0]], [j3[1], ee[1]])
        plt.draw()
        plt.title('Episode ' +  str(self._episode_no))
        plt.pause(.5)
        plt.clf()
