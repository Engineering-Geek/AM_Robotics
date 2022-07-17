from gym.envs.registration import register

register(
    id='AM_Robotics-v0',
    entry_point='gym_AM_Robotics.envs:AMRoboticsEnv',
    kwargs={
        'mjcf_path': '/home/nmelgiri/AM_Robotics/AM_Robot/arm.xml',
        'frame_skip': 1,
        'width': 64,
        'height': 64,
        'mujoco_bindings': "mujoco",
        'contact_reward': 50
    }
)
