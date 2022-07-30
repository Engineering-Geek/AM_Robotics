import gym
import gym_AM_Robotics
import multiprocessing
import logging
from gym.wrappers import Monitor, RecordVideo
import os

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv

class Trainer:
    def __init__(self, env_name: str, model_name: str, policy: str, 
                 model_path: str, mjcf_path: str, render_mode: str, frame_skip: int,
                 contact_reward: float, max_steps: int, 
                 total_timesteps: int, tb_log_name: str, n_cpu: int, seed: int):
        self.env_name = env_name
        self.model_name = model_name
        self.policy = policy
        self.model_path = model_path
        self.mjcf_path = mjcf_path
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.contact_reward = contact_reward
        self.max_steps = max_steps
        self.total_timesteps = total_timesteps
        self.tb_log_name = tb_log_name

        print("Creating environment using {} cpus...".format(n_cpu))        
        self.envs = self._env_creator(n_cpu=n_cpu, seed=seed)
        self.env = gym.make(
            id=self.env_name, 
            mjcf_path=self.mjcf_path, 
            render_mode=self.render_mode,
            frame_skip=self.frame_skip,
            contact_reward=self.contact_reward,
            max_steps=self.max_steps,
        )
        self.model = self._generate_model()
        
    def _env_generator(self, rank: int, seed=0):
        def _init():
            try:
                env = gym.make(
                    id=self.env_name, 
                    mjcf_path=self.mjcf_path, 
                    render_mode=self.render_mode,
                    frame_skip=self.frame_skip,
                    contact_reward=self.contact_reward,
                    max_steps=self.max_steps,
                )
            except Exception as e:
                env = gym.make(id=self.env_name)
            env.seed(seed + rank)
            return env
        return _init
    
    def _env_creator(self, n_cpu: int, seed=0):
        return DummyVecEnv([self._env_generator(i, seed) for i in range(n_cpu)])
    
    def _generate_model(self):
        if self.model_name == "A2C":
            return A2C(self.policy, self.envs, verbose=1, tensorboard_log=self.tb_log_name)
        elif self.model_name == "DQN":
            return DQN(self.policy, self.envs, verbose=1, tensorboard_log=self.tb_log_name)
        elif self.model_name == "PPO":
            return PPO(self.policy, self.envs, verbose=1, tensorboard_log=self.tb_log_name)
        elif self.model_name == "SAC":
            return SAC(self.policy, self.envs, verbose=1, tensorboard_log=self.tb_log_name)
        elif self.model_name == "TD3":
            # Note, TD3 only supports single process environments at the moment
            return TD3(self.policy, self.env, verbose=1, tensorboard_log=self.tb_log_name)
        else:
            raise ValueError("Model name {} is not supported. Valid names include: \
                            A2C, DQN, PPO, SAC, TD3".format(self.model_name))


    def close(self):
        self.envs.close()
    
    def save(self):
        print("saving model to {}".format(self.model_path))
        self.model.save(self.model_path)
    
    def load(self):
        self.model.load(self.model_path)
    
    def train(self):
        self.model.learn(total_timesteps=self.total_timesteps, tb_log_name=self.tb_log_name)
    
    def test(self, name: str):
        os.environ['MUJOCO_GL']="egl"
        env = RecordVideo(self.env, name)
        obs = env.reset()
        while True:
            action, _ = self.model.predict(obs)
            obs, _, done, _ = env.step(action)
            if done.any():
                obs = env.reset()
