from typing import Dict
from collections import deque

import numpy as np
import gymnasium as gym
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave


class BuildingEnv(gym.Env):
    def __init__(self, config: Dict):

        self.config = config
        self.start_time = config['start_time']
        self.end_time = config['stop_time']
        self.step_size = config['step_size']
        self.current_time = self.start_time

        self.input = list(config["input"].keys())
        self.output = list(config["output"].keys())
        self.observations = list(config["observations"].keys())

        # Buffer for lagged observations (t-1, t-2, t-3)
        self.temp_history = deque(maxlen=3)

        try:
            self.idx_zone_temp = self.observations.index("zone_temp")
            self.idx_outdoor = self.observations.index("weaBus.TDryBul")
        except ValueError:
            self.idx_zone_temp = 0
            self.idx_outdoor = 1

        self.model_description = read_model_description(config["fmu_path"])
        self.unzip_dir = extract(config["fmu_path"])
        self.fmu = None
        self.vrs = {}

        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(len(config["input"].keys()),), dtype=np.float32
        )

        # 1. Base Observations bounds
        low_obs_list = []
        high_obs_list = []
        for key in config["observations"].keys():
            low_obs_list.append(config["observations"][key]["min"])
            high_obs_list.append(config["observations"][key]["max"])

        # 2. Synthetic Observations bounds

        # Energy Price [0.0, 0.13]
        low_obs_list.append(0.0)
        high_obs_list.append(0.13)

        # Time to Occupancy Start [0, 24h]
        low_obs_list.append(0.0)
        high_obs_list.append(86400.0)

        # Time to Occupancy End [0, 24h]
        low_obs_list.append(0.0)
        high_obs_list.append(86400.0)

        # Delta Outdoor [-30, 60]
        low_obs_list.append(-30.0)
        high_obs_list.append(60.0)

        # Delta Zone [-20, 20]
        low_obs_list.append(-20.0)
        high_obs_list.append(20.0)

        # 3. Lagged Observations bounds (Same as zone_temp)
        # Lag 1
        low_obs_list.append(config["observations"]["zone_temp"]["min"])
        high_obs_list.append(config["observations"]["zone_temp"]["max"])
        # Lag 2
        low_obs_list.append(config["observations"]["zone_temp"]["min"])
        high_obs_list.append(config["observations"]["zone_temp"]["max"])
        # Lag 3
        low_obs_list.append(config["observations"]["zone_temp"]["min"])
        high_obs_list.append(config["observations"]["zone_temp"]["max"])

        self.obs_min = np.array(low_obs_list, dtype=np.float32)
        self.obs_max = np.array(high_obs_list, dtype=np.float32)

        self.obs_range = self.obs_max - self.obs_min
        self.obs_range[self.obs_range == 0] = 1.0

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.obs_min.shape,
            dtype=np.float32
        )

        self._map_variables()

    def _map_variables(self):

        for name in self.input:
            var = next(v for v in self.model_description.modelVariables if v.name == name)
            self.vrs[name] = var.valueReference

        for name in self.output:
            var = next(v for v in self.model_description.modelVariables if v.name == name)
            self.vrs[name] = var.valueReference

        for name in self.observations:
            var = next(v for v in self.model_description.modelVariables if v.name == name)
            self.vrs[name] = var.valueReference

    def normalize_observation(self, obs):
        return (obs - self.obs_min) / self.obs_range

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.fmu:
            self.fmu.terminate()
            self.fmu.freeInstance()

        self.current_time = self.start_time

        self.fmu = FMU2Slave(
            guid=self.model_description.guid,
            unzipDirectory=self.unzip_dir,
            modelIdentifier=self.model_description.coSimulation.modelIdentifier,
            instanceName='simpleHouse'
        )

        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=self.start_time)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

        # Initialize history with the starting temperature
        initial_raw_obs = self._get_base_fmu_values()
        initial_temp = initial_raw_obs[self.idx_zone_temp]

        self.temp_history.clear()
        for _ in range(3):
            self.temp_history.append(initial_temp)

        return self._get_observation(), {}

    def _get_base_fmu_values(self):
        list_vrs_observations = [self.vrs[obs] for obs in self.observations]
        return list(self.fmu.getReal(list_vrs_observations))

    def _get_raw_observation(self):
        # Get base values from FMU
        raw_values = self._get_base_fmu_values()

        # --- Calculations ---
        day_seconds = self.current_time % 86400
        hour = day_seconds / 3600.0

        # 1. Energy Price
        if 7.0 <= hour < 19.0:
            energy_price = 0.13
        else:
            energy_price = 0.13 * 0.3

        # 2. Time Logic (Schedule: 08:00 - 18:00)
        occ_start_sec = 28800  # 08:00
        occ_end_sec = 64800  # 18:00

        # Time to Occupancy Start
        if day_seconds < occ_start_sec:
            time_to_start = occ_start_sec - day_seconds
        elif day_seconds < occ_end_sec:
            time_to_start = 0.0
        else:
            time_to_start = (86400 - day_seconds) + occ_start_sec

        # Time to Occupancy End
        if occ_start_sec <= day_seconds < occ_end_sec:
            time_to_end = occ_end_sec - day_seconds
        else:
            time_to_end = 0.0

        # 3. Deltas
        t_out = raw_values[self.idx_outdoor]
        t_zone = raw_values[self.idx_zone_temp]

        delta_outdoor = 20.0 - t_out

        if occ_start_sec <= day_seconds < occ_end_sec:
            active_setpoint = 20.0
        else:
            active_setpoint = 15.0

        delta_zone = active_setpoint - t_zone

        # Append Synthetic Observations
        raw_values.append(energy_price)
        raw_values.append(time_to_start)
        raw_values.append(time_to_end)
        raw_values.append(delta_outdoor)
        raw_values.append(delta_zone)

        # Append Lagged Observations (t-1, t-2, t-3)
        raw_values.extend(list(self.temp_history))

        return np.array(raw_values, dtype=np.float32)

    def _get_observation(self):
        raw_obs = self._get_raw_observation()
        return self.normalize_observation(raw_obs)

    def step(self, action):

        list_vrs_input = [self.vrs[inpt] for inpt in self.input]
        self.fmu.setReal(list_vrs_input, action)
        self.fmu.doStep(currentCommunicationPoint=self.current_time, communicationStepSize=self.step_size)
        self.current_time += self.step_size

        # 1. Get observation (uses current history for lags)
        raw_obs = self._get_raw_observation()
        norm_obs = self.normalize_observation(raw_obs)

        # 2. Update history with the CURRENT temp for the NEXT step
        # raw_obs contains base vars at indices [0..len(observations)-1]
        current_temp = raw_obs[self.idx_zone_temp]
        self.temp_history.append(current_temp)

        obs_dict = {}
        for i, obs_name in enumerate(self.observations):
            obs_dict[obs_name] = raw_obs[i]

        # Use index relative to the raw_obs array construction
        base_len = len(self.observations)
        obs_dict["energy_price"] = raw_obs[base_len]

        reward_comf, reward_cost = self._calculate_reward(obs_dict)
        reward = reward_comf + reward_cost
        terminated = self.current_time >= self.end_time
        truncated = False

        return norm_obs, reward, terminated, truncated, obs_dict

    @staticmethod
    def _calculate_reward(observations):
        w_comfort = 1.6
        w_cost = 1

        zone_temp = observations.get("zone_temp")
        rad_heat = observations.get("rad_heat")
        occupancy = observations.get("booltoOcc.y")
        energy_price = observations.get("energy_price")

        penalty_comfort = 0.0

        if occupancy > 0.5:
            if zone_temp < 19.0:
                penalty_comfort = (20 - zone_temp) ** 2
            elif zone_temp > 21.0:
                penalty_comfort = (zone_temp - 20.0) ** 2
        else:
            if zone_temp < 17.0:
                penalty_comfort = (17.0 - zone_temp) ** 2

        cost_term = rad_heat * (5.0 / 12.0) * energy_price / 1000.0

        return - (penalty_comfort * w_comfort), - (w_cost * cost_term)

    def close(self):
        if self.fmu:
            self.fmu.terminate()
            self.fmu.freeInstance()