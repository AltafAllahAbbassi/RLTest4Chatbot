from abc import ABC, abstractmethod
from gym import Env

class Environment(Env, ABC):
    @abstractmethod
    def is_done(self):
        pass

    @abstractmethod
    def next_state(self):
        pass
    
    @abstractmethod
    def calculate_reward(self):
        pass

    @abstractmethod
    def set_state(self):
        pass




