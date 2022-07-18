import gym
import gym_AM_Robotics

from stable_baselines3 import A2C

env = gym.make(
    id="AM_Robotics-v0", 
    mjcf_path="/home/nmelgiri/AM_Robotics/AM_Robot/arm_target.xml", 
    render_mode="rgb_array",
    frame_skip=1,
    contact_reward=1e5,
    max_steps=1000
)

model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="./a2c_arm_tensorboard/")
model.learn(total_timesteps=500000, tb_log_name="updated_cr")
model.save("a2c_arm")

model = A2C.load("a2c_arm")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()