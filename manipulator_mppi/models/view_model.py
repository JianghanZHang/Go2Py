import os
import mujoco
from mujoco import viewer

def main():
    # ----------------------------------------------------------------------
    # 1) Define the path to your MuJoCo XML file
    # ----------------------------------------------------------------------
    # E.g., if view_model.py is in the same folder as "nyufinger/", do:
    # script_dir = os.path.dirname(os.path.realpath(__file__))
    # xml_path = os.path.join(script_dir, "nyufinger", "trifinger_nyu_scene.xml")
    #
    # Or just hard-code a relative or absolute path:
    xml_path = "nyufinger/finger_nyu_noselfcollision.xml"

    xml_path = "trifinger/finger.xml"

    # xml_path = "trifinger/trifinger_cube_scene_noselfcollision.xml"


    # ----------------------------------------------------------------------
    # 2) Load the MuJoCo model and create its data object
    # ----------------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # ----------------------------------------------------------------------
    # 3) Launch the interactive viewer
    # ----------------------------------------------------------------------
    viewer.launch(model, data)

if __name__ == "__main__":
    main()
