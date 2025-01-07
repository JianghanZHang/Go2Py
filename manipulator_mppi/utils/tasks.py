"""
Task definitions for robot navigation and behavior scenarios.

Each task is represented as a dictionary containing key parameters:
- `goal_pos`: List of target positions in the format [x, y, z].
- `default_orientation`: Default orientation of the robot as a quaternion [w, x, y, z].
- `cmd_vel`: Commanded velocities in the format [linear x, linear y] in body frame.
- `goal_thresh`: Thresholds for achieving goals.
- `desired_gait`: Gait type for each phase of the task.
- `waiting_times`: Time in milliseconds to wait at each phase.
- `model_path`: Path to the robot's model file.
- `config_path`: Path to the robot's configuration file.
- `sim_path`: Path to the simulation file.
"""

DEFAULT_MODEL_PATH = 'models/trifinger/trifinger_scene.xml'
DEFAULT_CONFIG_PATH = 'configs/mppi_trifinger_reaching.yml'
DEFAULT_SIM_PATH = 'models/trifinger/trifinger_scene.xml'
DEFAULT_ORIENTATION = [[1, 0, 0, 0]]


TASKS = {
    "reaching": {
        "finger_tips_pos": [[0.0, 0.0, 0.10],
                            [0.0, 0.0, 0.15],
                            [0.0, 0.0, 0.20]],
        "goal_thresh": [0.01, 
                        0.01, 
                        0.01],
        "waiting_times": [0, 
                          0, 
                          0],
        "model_path": DEFAULT_MODEL_PATH,
        "config_path": DEFAULT_CONFIG_PATH,
        "sim_path": DEFAULT_SIM_PATH
    }
}

def get_task(task_name):
    """
    Retrieve task configuration by name.

    Args:
        task_name (str): Name of the task. Must be one of the keys in TASKS.

    Returns:
        dict: Task configuration dictionary.

    Raises:
        ValueError: If the task_name is not found in TASKS.
    """
    if task_name not in TASKS:
        raise ValueError(f"Task '{task_name}' not found. Available tasks: {list(TASKS.keys())}")
    return TASKS[task_name]