import numpy as np


class ObservationPreprocessor:

    def __init__(self, observation_space):
        # Note: This assumes observation_space is a gym.spaces.Box object
        self.observation_min = observation_space.low
        self.observation_max = observation_space.high

    def process(self, observation):
        # Clamp observation to its valid range in case it goes beyond due to numerical issues
        observation = np.clip(observation, self.observation_min, self.observation_max)

        # Normalize observation to [0, 1]
        normalized_observation = (observation - self.observation_min) / (self.observation_max - self.observation_min)

        return normalized_observation


class ObservationPreprocessorIdentity:

    def __init__(self, observation_space):
        pass

    def process(self, observation):
        return observation
