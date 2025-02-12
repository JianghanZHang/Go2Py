import numpy as np
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
import mujoco
from scipy.interpolate import CubicSpline
from mujoco.rollout import Rollout
import yaml
from scipy.stats import qmc


class BaseMPPI:
    """
    Base class for Model Predictive Path Integral (MPPI) controllers.
    Provides shared functionality for all task-specific controllers.
    """

    def __init__(self, model_path, config_path):
        """
        Initialize common MPPI parameters and configurations.

        Args:
            params (dict): Dictionary of task parameters from configuration.
            model_path (str): Path to the MuJoCo model XML file.
            config_path (str): Path to the configuration file.
        """

        # Load task-specific configurations
        with open(config_path, 'r') as file:
            params = yaml.safe_load(file)

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = params['dt']
        self.model.opt.enableflags = 1  # Override contact settings
        self.model.opt.o_solref = np.array(params['o_solref'])

        # MPPI parameters
        self.temperature = params['lambda']
        self.horizon = params['horizon']
        self.n_samples = params['n_samples']
        self.noise_sigma = np.array(params['noise_sigma'])
        self.num_workers = params['n_workers']

        self.sampling_init = np.array([-0.3, 1.34, -2.83, 0.3, 1.34, -2.83] * 2)

        # Initialize rollouts and sampling configurations
        self.h = params['dt']
        self.sample_type = params['sample_type']
        self.n_knots = params['n_knots']
        self.random_generator = np.random.default_rng(params["seed"])
        self.rollout_func = self.threaded_rollout
        self.cost_func = self.calculate_total_cost

        # Threading
        # self.thread_local = threading.local()
        # self.executor = ThreadPoolExecutor(max_workers=self.num_workers, initializer=self.thread_initializer)
        self.parallel_rollout = Rollout(nthread=self.num_workers)


        # Initialize rollouts
        self.state_rollouts = np.zeros(
            (self.n_samples, self.horizon, mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value))
        )
        self.selected_trajectory = None

        # self.sensor_datas = np.zeros(
        #     (self.n_samples, self.horizon, self.sensor_data_size)
        # )
        
        self.rollout_models = [self.model for _ in range(self.n_samples)]
        self.rollout_model = [self.model]

        self.mujoco_data = [mujoco.MjData(self.model) for _ in range(self.num_workers)]

        self.selected_trajectory = None

        # Action limits
        self.act_dim = 12
        self.act_max = np.array([1.0472, 3.4907, -0.83776,   #FL
                                 1.0472, 3.4907, -0.83776,   #FR
                                 1.0472, 4.5379, -0.83776,   #RL
                                 1.0472, 4.5379, -0.83776,]) #RR

        self.act_min = np.array([-1.0472, -1.5708, -2.7227,
                                 -1.0472, -1.5708, -2.7227,
                                 -1.0472, -0.5236, -2.7227,
                                 -1.0472, -0.5236, -2.7227,])

        # <motor ctrlrange="-23.7 23.7"/>
        # <default class="abduction">
        # <joint axis="1 0 0" range="-1.0472 1.0472"/>
        # </default>
        # <default class="hip">
        # <default class="front_hip">

        # <joint range="-1.5708 3.4907"/>
        # </default>
        # <default class="back_hip">
        # <joint range="-0.5236 4.5379"/>
        # </default>
        # </default>

        # <default class="knee">
        # <joint range="-2.7227 -0.83776"/>
        # <motor ctrlrange="-45.43 45.43"/>

        # Noise
        self.noise_type = params['noise_type']

        if self.noise_type == 'gaussian':
            self.random_generator = np.random.default_rng(params["seed"])
            self.generate_noise = self.generate_Gaussian

        if self.noise_type == 'halton':
            self.random_generator = qmc.Halton(d = self.act_dim*self.n_knots, scramble=True, seed=params["seed"])
            self.generate_noise = self.generate_Halton

    def reset_planner(self):
        """Reset the action planner to its initial state."""
        self.trajectory = np.zeros((self.horizon, self.act_dim))
        self.trajectory += self.sampling_init

    def sample_delta_u(self):
        if self.sample_type == 'normal':
            size = (self.n_samples, self.horizon, self.act_dim)
            return self.generate_noise(size)
        elif self.sample_type == 'cubic':
            indices = np.arange(self.n_knots)*self.horizon//self.n_knots
            size = (self.n_samples, self.n_knots, self.act_dim)
            knot_points = self.generate_noise(size)
            cubic_spline = CubicSpline(indices, knot_points, axis=1)
            return cubic_spline(np.arange(self.horizon))
        
    def perturb_action(self):
        if self.sample_type == 'normal':
            size = (self.n_samples, self.horizon, self.act_dim)
            actions = self.trajectory + self.generate_noise(size)
            actions = np.clip(actions, self.act_min, self.act_max)
            return actions
        
        elif self.sample_type == 'cubic':
            indices_float = np.linspace(0, self.horizon - 1, num=self.n_knots)
            indices = np.round(indices_float).astype(int)
            size = (self.n_samples, self.n_knots, self.act_dim)
            noise = self.generate_noise(size)
            knot_points = self.trajectory[indices] + noise
            #knot_points[:, 0, :] = self.trajectory[0]
            cubic_spline = CubicSpline(indices, knot_points, axis=1)
            actions = cubic_spline(np.arange(self.horizon))
            actions = np.clip(actions, self.act_min, self.act_max)
            return actions
        
    def generate_Gaussian(self, size):
        """
        Generate noise for sampling actions.

        Args:
            size (tuple): Shape of the noise array.

        Returns:
            np.ndarray: Generated noise scaled by `noise_sigma`.
        """
        return self.random_generator.normal(size=size) * self.noise_sigma

    
    def generate_Halton(self, size):
        """
        Generate noise for sampling actions using a Halton sequence.
        The result is shape=(n_samples, horizon, act_dim), similar to generate_Gaussian.

        Args:
            size (tuple): (n_samples, horizon, act_dim)
            noise_sigma (np.ndarray): Per-dimension scaling of noise, shape=(act_dim,)

        Returns:
            np.ndarray: Halton "noise" in the same shape as `generate_Gaussian`,
                        scaled by `noise_sigma`, range approximately [-1, 1].
        """
        n_samples, horizon, act_dim = size

        # 1) We treat the dimension as (horizon * act_dim).
        # dimension = horizon * act_dim

        # 2) Create or re-use a Halton sampler.
        # In the constructor

        # 3) Generate n_samples points in [0,1]^dimension
        #    => shape (n_samples, dimension)
        # Here this random_generator is a Halton sequence of size (self.act_dim*self.n_knots)
        halton_2d = self.random_generator.random(n=n_samples)

        # 4) Reshape to (n_samples, horizon, act_dim)
        halton_3d = halton_2d.reshape(n_samples, horizon, act_dim)

        # 5) Shift from [0,1] to [-0.5,0.5], then scale up by factor 2 => [-1,1]
        halton_3d = (halton_3d - 0.5) * 2.0

        # 6) Multiply by noise_sigma for each action dimension
        #    noise_sigma has shape (act_dim,) => broadcast along (n_samples, horizon)
        halton_3d *= self.noise_sigma  # shape => (n_samples, horizon, act_dim)

        return halton_3d

    # def thread_initializer(self):
    #     """Initialize thread-local storage for MuJoCo data."""
    #     self.thread_local.data = mujoco.MjData(self.model)

    # def shutdown(self):
    #     """Shutdown the thread pool executor."""
    #     self.executor.shutdown(wait=True)

    # def call_rollout(self, initial_state, ctrl, state):
    #     """
    #     Perform a rollout of the model given the initial state and control actions.

    #     Args:
    #         initial_state (np.ndarray): Initial state of the model.
    #         ctrl (np.ndarray): Control actions to apply during the rollout.
    #         state (np.ndarray): State array to store the results of the rollout.
    #     """
    #     # rollout.rollout(self.model, self.thread_local.data, skip_checks=True,
    #     #                 nroll=state.shape[0], nstep=state.shape[1],
    #     #                 initial_state=initial_state, control=ctrl, state=state)

    #     # see https://mujoco.readthedocs.io/en/latest/changelog.html#id1 for changes in rollout function
    #     rollout.rollout(self.model, self.thread_local.data, skip_checks=False, 
    #                     nstep=state.shape[1], initial_state=initial_state, control=ctrl, state=state)

    # def threaded_rollout(self, state, ctrl, initial_state, num_workers=32, nstep=5):
    #     """
    #     Perform rollouts in parallel using a thread pool.

    #     Args:
    #         state (np.ndarray): Array to store the results of the rollouts.
    #         ctrl (np.ndarray): Control actions for the rollouts.
    #         initial_state (np.ndarray): Initial states for the rollouts.
    #         num_workers (int): Number of parallel threads to use.
    #         nstep (int): Number of steps in each rollout.
    #     """
    #     n = len(initial_state) // num_workers

    #     # Divide tasks into chunks for each worker
    #     chunks = [(initial_state[i * n:(i + 1) * n], ctrl[i * n:(i + 1) * n], state[i * n:(i + 1) * n])
    #             for i in range(num_workers - 1)]

    #     # Add remaining chunk
    #     chunks.append((initial_state[(num_workers - 1) * n:], ctrl[(num_workers - 1) * n:], state[(num_workers - 1) * n:]))

    #     # Submit tasks to thread pool
    #     futures = [self.executor.submit(self.call_rollout, *chunk) for chunk in chunks]
    #     for future in concurrent.futures.as_completed(futures):
    #         future.result()  # Ensure all threads complete execution


    def threaded_rollout(self, model, state, ctrl, initial_state, **kwargs):
        """
        Perform rollouts in parallel using MuJoCo's native batched rollout.

        This function uses MuJoCo's rollout API from the official documentation.
        It creates a list of MjData objects (one per thread) so that the native rollout
        uses internal multithreading. Extra keyword arguments (like nstep) are passed along.

        Args:
            state (np.ndarray): Preallocated state output array of shape (nbatch, nstep, nstate).
            ctrl (np.ndarray): Control array of shape (nbatch, nstep, ncontrol).
            initial_state (np.ndarray): Initial state array of shape (nbatch, nstate).
            sensor_data (np.ndarray): Preallocated sensor data array of shape (nbatch, nstep, nsensordata).
            **kwargs: Additional keyword arguments (e.g., nstep).
        """
        # Determine number of steps from kwargs or use horizon.
        # nstep = kwargs.get("nstep", self.horizon)
        # If self.num_workers > 0, we create that many MjData objects as in self.mujoco_data.
        
        
        # Call MuJoCo's native rollout function.
        # According to Mujoco's documentation, the function returns (state, sensordata),
        # Refer to this code: https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/rollout.py
        self.parallel_rollout.rollout(
            model= model,  # wrap the model in a list
            data=self.mujoco_data,
            initial_state=initial_state,
            nstep=state.shape[1],
            initial_warmstart=None,
            control=ctrl,
            skip_checks=True,
            control_spec=mujoco.mjtState.mjSTATE_CTRL.value,
            state=state,
            chunk_size=None
        )


    def set_params(self, horizon, lambda_, N):
        """
        Update MPPI parameters and reset controller.

        Args:
            horizon (int): Time horizon.
            lambda_ (float): Temperature parameter for MPPI.
            N (int): Number of samples for MPPI rollouts.
        """
        self.horizon = horizon
        self.temperature = lambda_
        self.n_samples = N

        # Reset state rollouts with updated dimensions
        self.state_rollouts = np.zeros(
            (self.n_samples, self.horizon, mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value))
        )

        # Reset the planner to its initial state
        self.reset_planner()

    # def __del__(self):
    #     self.shutdown()