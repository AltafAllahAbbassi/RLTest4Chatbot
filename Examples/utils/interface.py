from abc import ABC, abstractmethod

class ModelInterface(ABC):
    @abstractmethod
    def dst_query(self, dialogue: dict, turn_id: int, user_input: str):
        pass

    @abstractmethod
    def gini_query(self, dialogue: dict, turn_id: int, user_input: str):
        pass

    @abstractmethod
    def dst_gini_query(self, dialogue: dict, turn_id: int, user_input: str):
        pass
