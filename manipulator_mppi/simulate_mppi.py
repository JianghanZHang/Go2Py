import argparse
import os

# Adjust these imports to match where you have placed the trifinger code:
from interface.simulator import Simulator
from control.controllers.mppi_reaching import reaching_MPPI
from utils.tasks import get_task
import faulthandler

def main(task):
    # ---------------------------
    # Simulation and Controller Parameters
    # ---------------------------
    T = 2000  # total steps, e.g. 20 seconds if dt=0.01
    VIEWER = True

    SIMULATION_STEP = 0.01
    CTRL_UPDATE_RATE = 100     # control update frequency
    CTRL_HORIZON = 40
    CTRL_LAMBDA = 0.5
    CTRL_N_SAMPLES = 50

    # Soft contact model parameters
    TIMECONST = 0.02
    DAMPINGRATIO = 1.0

    # ---------------------------
    # Get trifinger-specific task data
    # ---------------------------
    # For example, tasks might be { "reaching": {"sim_path": "..."} }
    task_data = get_task(task)
    sim_path = task_data["sim_path"]  # path to your trifinger xml

    # ---------------------------
    # Initialize MPPI and simulator
    # ---------------------------
    agent = reaching_MPPI(task=task)
    agent.set_params(horizon=CTRL_HORIZON,
                     lambda_=CTRL_LAMBDA,
                     N=CTRL_N_SAMPLES)

    simulator = Simulator(
        agent=agent,
        viewer=VIEWER,
        T=T,
        dt=SIMULATION_STEP,
        timeconst=TIMECONST,
        dampingratio=DAMPINGRATIO,
        model_path=sim_path,
        ctrl_rate=CTRL_UPDATE_RATE
    )

    # ---------------------------
    # Run simulation + plotting
    # ---------------------------
    simulator.run()
    simulator.plot_trajectory()


if __name__ == "__main__":

    faulthandler.enable()

    # Example trifinger tasks:

    VALID_TASKS = ["reaching", "push_box"]

    parser = argparse.ArgumentParser(description="Run trifinger MPPI simulation.")
    parser.add_argument('--task',
                        type=str,
                        required=True,
                        choices=VALID_TASKS,
                        help=f"Name of the task. Must be one of: {VALID_TASKS}")
    args = parser.parse_args()

    main(args.task)
