import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import yaml
from mujoco import rollout
from scipy.interpolate import CubicSpline
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
        # self.model.opt.o_solref = np.array(params['o_solref'])

        # MPPI parameters
        self.temperature = params['lambda']
        self.horizon = params['horizon']
        self.n_samples = params['n_samples']
        self.noise_sigma = np.array(params['noise_sigma'])
        self.num_workers = params['n_workers']
        self.beta = params['beta']

        self.sensor_data_size = params['sensor_data_size']

        print(f'sensor_data_size:{self.sensor_data_size}')

        # self.sampling_init = np.array([0.0, -0.6, -1.2] * 3)

        self.sampling_init = np.array(self.model.key_qpos[0, :9])

        self.q_cube = [0.0, 0.0, 0.03,
                       1.0, 0.0, 0.0, 0.0]

        ##########  DEBUG   ############
        # import mujoco.viewer as viewer
        # Mjdata = mujoco.MjData(self.model)
        # Mjdata.qpos = np.hstack((self.sampling_init, self.q_cube))
        # mujoco.mj_forward(self.model, Mjdata)

        # # viewer.launch(self.model, Mjdata)

        # import pdb; pdb.set_trace()


        # Initialize rollouts and sampling configurations
        self.h = params['dt']
        self.sample_type = params['sample_type']
        self.n_knots = params['n_knots']
        self.rollout_func = self.threaded_rollout
        self.cost_func = self.calculate_total_cost

        # Threading
        self.thread_local = threading.local()
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers, initializer=self.thread_initializer)

        # Initialize rollouts
        self.state_rollouts = np.zeros(
            (self.n_samples, self.horizon, mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value))
        )

        # import pdb; pdb.set_trace()
        self.sensor_datas = np.zeros(
            (self.n_samples, self.horizon, self.sensor_data_size)
        )

        self.selected_trajectory = None

        # Action limits
        self.act_dim = 9

        self.act_min = np.array([-1.570796, -1.570796, -3.1415926] * 3)

        self.act_max = np.array([1.570796, 3.1415926, 3.1415926] * 3)

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
            actions = self.trajectory + self.generate_noise(size, self.noise_sigma)
            actions = np.clip(actions, self.act_min, self.act_max)
            return actions

        elif self.sample_type == 'cubic':
            indices_float = np.linspace(0, self.horizon - 1, num=self.n_knots)
            indices = np.round(indices_float).astype(int)
            size = (self.n_samples, self.n_knots, self.act_dim)
            noise = self.generate_noise(size, self.noise_sigma)
            filtered_noise = np.zeros_like(noise)
            filtered_noise[:, 0, :] = self.beta * noise[:, 0, :]

            # Smoother noise from Sergey Levine's paper: https://arxiv.org/pdf/1909.11652

            filtered_noise[:, 0, :] = noise[:, 0, :]

            for n in range(1, self.n_knots):
                filtered_noise[:, n, :] = self.beta * noise[:, n, :] + (1-self.beta) * filtered_noise[:, n-1, :]

            # assert (filtered_noise[0, 1, :] == self.beta * noise[0, 1, :]  +  (1-self.bseta) * filtered_noise[0, 0, :]).all()

            knot_points = self.trajectory[indices] + filtered_noise
            cubic_spline = CubicSpline(indices, knot_points, axis=1)
            actions = cubic_spline(np.arange(self.horizon))
            actions = np.clip(actions, self.act_min, self.act_max)
            return actions

    def generate_Gaussian(self, size, noise_sigma):
        """
        Generate noise for sampling actions.

        Args:
            size (tuple): Shape of the noise array.

        Returns:
            np.ndarray: Generated noise scaled by `noise_sigma`.
        """
        return self.random_generator.normal(size=size) * noise_sigma

    def generate_Halton(self, size, noise_sigma):
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
        halton_3d *= noise_sigma  # shape => (n_samples, horizon, act_dim)

        return halton_3d

    def thread_initializer(self):
        """Initialize thread-local storage for MuJoCo data."""
        self.thread_local.data = mujoco.MjData(self.model)

    def shutdown(self):
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=True)

    def call_rollout(self, initial_state, ctrl, state, sensor_data):
        """
        Perform a rollout of the model given the initial state and control actions.

        Args:
            initial_state (np.ndarray): Initial state of the model.
            ctrl (np.ndarray): Control actions to apply during the rollout.
            state (np.ndarray): State array to store the results of the rollout.
        """
        # rollout.rollout(self.model, self.thread_local.data, skip_checks=True,
        #                 nroll=state.shape[0], nstep=state.shape[1],
        #                 initial_state=initial_state, control=ctrl, state=state)

        # see https://mujoco.readthedocs.io/en/latest/changelog.html#id1 for changes in rollout function

        rollout.rollout(
        model=self.model,
        data=self.thread_local.data,
        # nstep=state.shape[1],        # horizon
        initial_state=initial_state, # shape (N, ) or (Nq+Nv, ) or something else
        control=ctrl,               # shape (nstep, nu)
        state=state,                # shape (nstep+1, state_dim)
        sensordata=sensor_data      # shape (nstep, nsensordata)
        )

    def threaded_rollout(self, state, ctrl, initial_state, sensor_data, num_workers=32, nstep=5):
        """
        Perform rollouts in parallel using a thread pool.

        Args:
            state (np.ndarray): Array to store the results of the rollouts.
            ctrl (np.ndarray): Control actions for the rollouts.
            initial_state (np.ndarray): Initial states for the rollouts.
            num_workers (int): Number of parallel threads to use.
            nstep (int): Number of steps in each rollout.
        """


        n = len(initial_state) // num_workers

        actual_workers = min(num_workers, len(initial_state))
        if actual_workers == 0:
            # Means there's no data or zero rollouts to run
            print('No avaiable worker!')
            return
        # Divide tasks into chunks for each worker

        chunks = [(initial_state[i * n:(i + 1) * n], ctrl[i * n:(i + 1) * n], state[i * n:(i + 1) * n], sensor_data[i * n:(i + 1) * n])
                for i in range(num_workers - 1)]


        # import pdb; pdb.set_trace()
        # corrected code
        chunks.append((initial_state[(num_workers - 1) * n:],
                    ctrl[(num_workers - 1) * n:],
                    state[(num_workers - 1) * n:],
                    sensor_data[(num_workers - 1) * n:]))

        print("initial_state shape:", initial_state.shape)
        print("ctrl shape:", ctrl.shape)
        print("state shape:", state.shape)
        print("sensor_data shape:", sensor_data.shape)
        print("num_workers:", num_workers)

        # Submit tasks to thread pool
        futures = [self.executor.submit(self.call_rollout, *chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Ensure all threads complete execution

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

        self.sensor_datas = np.zeros(
            (self.n_samples, self.horizon, self.sensor_data_size)
        )

        # Reset the planner to its initial state
        self.reset_planner()

    def __del__(self):
        self.shutdown()
