import gym
from gym import spaces
from gym.utils import seeding
import mujoco_py
import numpy as np


class TargetBallEnv(gym.Env):
    def __init__(self, mjcf_path):
        self.model = mujoco_py.load_model_from_path(mjcf_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = mujoco_py.MjViewer(self.sim)
        self.t = 0

        self.joint_names = ["base", "lower", "upper"]
        self.joint_indices = [self.model.get_joint_qpos_addr(joint) for joint in self.joint_names]
        self.target_index = self.model.get_joint_qpos_addr("target")

        self.state = None
        self.observation_space = gym.spaces.Dict(
            {
                "joint_angles": gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float),
                "target_pos": gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float),
            }
        )
        self.action_space = gym.spaces.Box(low=-3.14, high=3.14, shape=(3,))
        self.history = {
            "base": [],
            "lower": [],
            "upper": [],
        }

    def save(self, filename):
        self.sim.save(open(filename, "w"))

    def _get_tip_pos(self):
        return self.sim.data.get_body_xpos("upper")

    def _get_target_pos(self):
        return self.sim.data.get_body_xpos("target")

    def reward(self):
        return NotImplementedError()

    def step(self, action):
        pass

    def target_velocity(self):
        """
        Given the operable range of the robot arm, generate a random velocity vector such that the target will intersect
        the robot arm
        """
        pass

    def reset(self, seed=None, return_info=None, **kwargs):
        pass

    def render(self):
        self.viewer.render()
        return self.observation_space.sample()

