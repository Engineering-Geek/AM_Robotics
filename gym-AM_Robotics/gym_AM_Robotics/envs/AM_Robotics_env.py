from gym.envs.mujoco import mujoco_env
import mujoco
from gym import utils
import numpy as np


class AMRoboticsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, mjcf_path, frame_skip=1, mujoco_bindings="mujoco", render_mode="human", camera=False, width=64, height=64):
        self.width = width
        self.height = height
        self.camera = camera
        self.render_mode = render_mode
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(
            self,
            model_path=mjcf_path,
            frame_skip=frame_skip,
            mujoco_bindings=mujoco_bindings,
        )

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        observation = self._get_obs()
        return observation, 0, False, {}

    def viewer_setup(self):
        # Position the camera
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 10
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0.5
        pass

    def reset_model(self):
        # Reset model to original state.
        # This is called in the overall env.reset method
        # do not call this method directly.
        new_qpos = self.init_qpos + self.np_random.uniform(low=-1., high=1., size=self.model.nq,)
        new_qvel = self.init_qvel + self.np_random.randn(self.model.nv,) * 0.5
        self.set_state(new_qpos, new_qvel)
        return self._get_obs()

    def render(self, mode="human", camera_id=None, camera_name="camera"):
        return super().render(mode=mode, width=self.width, height=self.height, camera_id=camera_id, camera_name=camera_name)

    def _get_obs(self):
        # Observation of environment fed to agent. This should never be called
        # directly but should be returned through reset_model and step
        joint_angles = self.data.qpos.copy().astype(np.float)[:3]
        joint_velocities = self.data.qvel.copy().astype(np.float)[:3]
        joint_accelerations = self.data.qacc.copy().astype(np.float)[:3]
        if self.camera:
            camera = self.render(self.render_mode).astype(np.float).flatten()
            observation = [
                camera,
                joint_angles,
                joint_velocities,
                joint_accelerations,
            ]
        else:
            observation = [
                joint_angles,
                joint_velocities,
                joint_accelerations,
            ]
        return np.concatenate(observation).ravel()
