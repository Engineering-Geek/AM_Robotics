import mujoco_py
import os
import shutil


def urdf_to_mjb(urdf_path):
    """
    Converts an urdf file to a mjcf file.

    Parameters
    ----------
    urdf_path: str
        Path to the urdf file.

    Returns
    -------
    mjcf_path: str
        Path to the mjcf file.
    """
    filename = os.path.basename(urdf_path).split('.')[0]
    temp_xml = os.path.join(os.path.dirname(urdf_path), "temp.xml")
    mjcf_xml = os.path.join(os.path.dirname(urdf_path), "{}.xml".format(filename))
    urdf_xml = os.path.join(urdf_path)
    shutil.copyfile(urdf_xml, temp_xml)
    model = mujoco_py.load_model_from_path(temp_xml)
    sim = mujoco_py.MjSim(model)
    sim.save(open(mjcf_xml, "w"))
    os.remove(temp_xml)
    return mjcf_xml


if __name__ == "__main__":
    mjcf_xml = urdf_to_mjb(urdf_path="/home/nmelgiri/AM_Robotics/AM_Robot/arm.urdf")

