import gym
import gym_AM_Robotics
import numpy as np


if __name__ == "__main__":
    env = gym.make("AM_Robotics-v0", mjcf_path="/home/nmelgiri/AM_Robotics/AM_Robot/arm.xml", render_mode="rgb_array")
    obs = env.reset()
    print(env.observation_space)
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


