import gym
import gym_AM_Robotics
import os
from gym import wrappers
from gym.wrappers.monitoring.video_recorder import VideoRecorder
os.environ['MUJOCO_GL']="egl"

env = gym.make("AM_Robotics-v0", render_mode="rgb_array")
video_path = "./videos/test.mp4"
recorder = VideoRecorder(env, video_path)

for episode in range(2):
    observation = env.reset()
    step = 0
    total_reward = 0

    while True:
        step += 1
        recorder.capture_frame()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print("Episode: {0},\tSteps: {1},\tscore: {2}"
                  .format(episode, step, total_reward)
            )
            break
env.close()
recorder.close()