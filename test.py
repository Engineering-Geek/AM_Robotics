import gym
import gym_AM_Robotics


if __name__ == "__main__":
    env = gym.make(
        id="AM_Robotics-v0", 
        mjcf_path="/home/nmelgiri/AM_Robotics/AM_Robot/arm_target.xml", 
        render_mode="rgb_array",
        frame_skip=1,
        contact_reward=50,
        max_steps=2000
    )
    obs = env.reset()
    
    while True:
        # action = env.action_space.sample()
        action = [100, 0, 0]
        obs, reward, done, info = env.step(action)
        print(reward)
        env.render()
        if done:
            obs = env.reset()





