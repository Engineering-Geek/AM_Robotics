import gym
import gym_AM_Robotics
import numpy as np
from time import sleep


if __name__ == "__main__":
    env = gym.make(
        id="AM_Robotics-v0", 
        mjcf_path="/home/nmelgiri/AM_Robotics/AM_Robot/arm_target.xml", 
        render_mode="rgb_array",
        frame_skip=1,
        contact_reward=50,
    )
    obs = env.reset()
    print(env.action_space, env.observation_space)





