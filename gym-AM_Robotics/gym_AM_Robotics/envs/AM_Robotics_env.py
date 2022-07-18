from gym.envs.mujoco import mujoco_env
from gym import utils
import numpy as np


class AMRoboticsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, mjcf_path, frame_skip=1, render_mode="human", 
                 camera=False, width=64, height=64, contact_reward=50.0, max_steps=1000):
        self.width = width
        self.height = height
        self.camera = camera
        self.render_mode = render_mode
        self.min_distance = 5.0 # meters
        self.max_distance = 8.0 # meters
        self.qvel_max = 5.0
        self.arm_length = 1.0
        self.max_vel = 8.0
        self.c = [-.45, .75, -.6]
        self.done = False
        self.contact_reward = contact_reward
        self.max_steps = max_steps
        self.i = 0
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(
            self,
            model_path=mjcf_path,
            frame_skip=frame_skip
        )

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        if self.i >= self.max_steps:
            self.done = True
        self.i += 1
        observation = self._get_obs()
        return observation, self.reward(), self.done, {}

    def viewer_setup(self):
        # Position the camera
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 10
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0.5

    def _get_tip_pos(self):
        return self.data.xipos[3].copy()

    def _get_target_pos(self):
        return self.data.xipos[4].copy()
    
    @staticmethod
    def polar_to_cartesian(theta: float, phi: float, r: float):
        """Given $$\theta, \phi, r$$, return the corresponding cartesian coordinates

        Args:
            theta (float): _description_
            phi (float): _description_
            r (float): _description_

        Returns:
            _type_: _description_
        """
        return np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])
    
    @staticmethod
    def distance(a: np.array, b: np.array):
        assert len(a) == len(b), "a and b are not the same length"
        return np.sqrt(sum([(a_ - b_) ** 2 for a_, b_ in zip(a, b)]))
        
    def _get_new_ball_params(self):
        """Returns a dictionary containing new set of parameters for the ball to be tossed

        Returns:
            dict: ball spawn position, ball spawn velocity vector
        """
        v_mag = 1e10
        while np.isnan(v_mag) or v_mag > self.max_vel:
            # target qpos (quaternion) and qvel (dx, dy, dz)
            r = np.random.rand() * (self.max_distance - self.min_distance) + self.min_distance
            theta = np.random.rand() * np.pi - np.pi / 2
            phi = np.random.rand() * 2 * np.pi
            ball_rel_pos = self.polar_to_cartesian(theta, phi, r)
            
            # pick a target within the range of the arm to throw it at that point (aim)
            theta1 = np.random.rand() * np.pi
            phi1 = np.random.rand() * np.pi
            r1 = np.random.rand() * self.arm_length
            aim_pos = self.polar_to_cartesian(theta1, phi1, r1)
            
            # pick a theta and find velocity, velocity eqn is derived from kinematic equations
            theta2 = np.random.rand() * np.pi / 4
            x = np.sqrt((ball_rel_pos[0] - aim_pos[0]) ** 2 + (ball_rel_pos[1] - aim_pos[1]) ** 2 )
            y = ball_rel_pos[1] - aim_pos[1]
            g = -9.81
            v_mag = np.sqrt(g * x ** 2 / ((2 * np.cos(theta2) ** 2) * (x * np.tan(theta2) - y)))

            if not np.isnan(v_mag):
                xy_vector = np.add(aim_pos, ball_rel_pos * -1)[:2]
                xy_unit_vector = xy_vector / np.linalg.norm(xy_vector)
                phi2 = np.arctan(xy_unit_vector[1] / xy_unit_vector[0])
                v_xy_magnitude = v_mag * np.cos(theta2)
                v_x = v_xy_magnitude * np.cos(phi2)
                v_y = v_xy_magnitude * np.sin(phi2)
                v_z = v_mag * np.cos(theta2)
                v = np.array([v_x, v_y, v_z])
                # double check to see if vx and vy approach target
                # project vector(x,y) v onto xy_vector
                v_xy = v[:2]
                projection = np.dot(v_xy, xy_vector) / np.linalg.norm(xy_vector)
                if projection < 0:
                    v *= -1
        return {
            "ball_spawn_position": np.add(ball_rel_pos, self.c),
            "ball_spawn_velocity": v,
        }
        
    def reset_model(self):
        # Reset model to original state.
        # This is called in the overall env.reset method
        # do not call this method directly.
        
        # arm qpos and qvel
        arm_qpos = self.init_qpos[:3] + self.np_random.uniform(low=-1., high=1., size=3,)
        arm_qvel = self.init_qvel[:3] + self.np_random.randn(3,) * self.qvel_max
        
        # ball qpos and qvel
        ball_params = self._get_new_ball_params()
        ball_qpos = ball_params["ball_spawn_position"]
        ball_qvel = ball_params["ball_spawn_velocity"]
        
        # joining them together and filling the remaining space with 0s (as those places are reserved for rotation quaternion and 3d angular velocity)
        qpos = np.concatenate([arm_qpos, ball_qpos, np.zeros((4,))])
        qvel = np.concatenate([arm_qvel, ball_qvel, np.zeros((3,))])
        self.set_state(qpos=qpos, qvel=qvel)
        self.done = False
        self.i = 0
        return self._get_obs()

    def render(self, mode="human", camera_id=None, camera_name="camera"):
        return super().render(mode=mode, width=self.width, height=self.height, camera_id=camera_id, camera_name=camera_name)

    def _get_obs(self):
        # Observation of environment fed to agent. This should never be called
        # directly but should be returned through reset_model and step
        joint_angles = self.data.qpos.copy().astype(float)[:3]
        joint_velocities = self.data.qvel.copy().astype(float)[:3]
        joint_accelerations = self.data.qacc.copy().astype(float)[:3]
        tip_pos = self._get_tip_pos()
        target_pos = self._get_target_pos()
        target_vector = np.add(target_pos, -tip_pos)
        if self.camera:
            camera = self.render(self.render_mode).astype(float).flatten()
            observation = [
                camera,
                joint_angles,
                joint_velocities,
                joint_accelerations,
                target_vector
            ]
        else:
            observation = [
                joint_angles,
                joint_velocities,
                joint_accelerations,
                target_vector
            ]
        return np.concatenate(observation).ravel()
    
    @staticmethod
    def str_mj_arr(arr):
        return ' '.join(['%0.3f' % arr[i] for i in range(len(arr))])

    def reward(self):
        # higher reward the closer the upper arm of the arm is to the ball
        distance = np.linalg.norm(self._get_target_pos() - self._get_tip_pos())
        reward = 1 / (100 * distance)
        
        # check to see if the ball hit the upper arm, if so, end episode with increased reward
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.contact[coni]
            if (con.geom1 == 3 or con.geom1 == 2) and (con.geom2 == 2 or con.geom2 == 3):
                self.done = True
                reward += self.contact_reward
        return reward

