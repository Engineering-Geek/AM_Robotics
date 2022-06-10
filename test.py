import numpy as np
from collections import OrderedDict

obs = OrderedDict(
    camera=np.zeros((64, 64, 3)),
    joint_angles=np.zeros(7),
    joint_velocities=np.zeros(7),
)

# What should happen there:
if any(np.any(np.isnan(obs[key])) for key in obs):
    print(any(np.any(np.isnan(obs[key])) for key in obs))
    raise ValueError("Observation contains NaN values")

print(np.any(np.isnan(obs)))





