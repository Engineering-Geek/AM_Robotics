from gym.envs.registration import register

register(
    id='AM_Robotics-v0',
    entry_point='gym_AM_Robotics.envs:AMRoboticsEnv',
    kwargs={
        'mjcf_path': '/home/root2/AM_Robotics/AM_Robot/arm_target.xml',
        'frame_skip': 1,
        'width': 64,
        'height': 64,
        'contact_reward': 50,
        'max_steps': 1000
    }
)
