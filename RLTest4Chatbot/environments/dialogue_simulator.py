"""
Questions and todos
The environment is too coupled with the dataset Multiwoz :/. I don't know what I should do
To 

"""

from gym import spaces
from RLTest4Chatbot.environments.environment import Environment
from RLTest4Chatbot.environments.utils.constants import STATE_ELEMENTS, DIALOG_POS, TURN_POS, MAX_WORDS, VALID_RATE, TRANSFORMATIONS, OBSERVATION_LOWER, OBSERVATION_UPPER
from RLTest4Chatbot.transformation.transformer import CompoundTransformer
from RLTest4Chatbot.transformation.helpers import calculate_modif_rate
# from Examples.trade.interface4trade import TradeInterface
from Examples.MultiWOZ.util import build_dict
import random
import numpy as np

class DialogueSimulator(Environment):
    """
    Dialogue Simulator with parametric actions
    """
    def __init__(self, data_file, model_interface):
        super().__init__()
        self.DIALOG_POS = DIALOG_POS
        self.TURN_POS = TURN_POS
        self.STATE_ELEMENTS =  STATE_ELEMENTS
        self.observation_shape = (self.STATE_ELEMENTS,)
        self.compound_transfomer = CompoundTransformer(TRANSFORMATIONS)
        self.ACTIONS = self.compound_transfomer.get_actions()

        self.interface = model_interface()
        self.data_file = data_file
        self.data_maps, _, self.dialogues = build_dict(self.data_file) 

        self.state =  self.reset()
        self.num_actions = len(list(self.ACTIONS.keys()))
        self.n_hidden_action = round(MAX_WORDS*VALID_RATE) 

        self.action_space = spaces.Tuple((
            spaces.Discrete(self.num_actions),
            *spaces.Tuple(  # parameters
                tuple(spaces.Box(low=np.zeros(self.ACTIONS[i][1]*self.n_hidden_action, dtype= "float"), high=np.ones(self.ACTIONS[i][1]*self.n_hidden_action, dtype= "float"), dtype=np.float32)
                      for i in range(self.num_actions))
            )
        ))

        # multi discrete action space
        self.observation_space = spaces.Tuple((spaces.Box(shape=self.observation_shape,
                                                          low = OBSERVATION_LOWER,
                                                          high= OBSERVATION_LOWER
                                                          ),
                                               ))
        self.action_parameter_sizes = np.array([self.ACTIONS[i][1]*self.n_hidden_action for i in range(self.num_actions)])
        self.action_parameter_size = self.action_parameter_sizes.sum()
        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)

    def set_data_file(self, data_file):
        self.data_file = data_file
        self.data_maps, _, self.dialogues = build_dict(data_file)
        self.state = self.reset()
        
    def reset(self):
        state = [None] * STATE_ELEMENTS
        state[self.DIALOG_POS] = random.choice(
            list(range(len(self.dialogues))))    # I don't know : the choice is random, keep it or change this
        state[self.TURN_POS] = 0
        return state

    def render(self):
        pass

    def close(self):
        pass

    def is_done(self):
        dialog_index = self.dialogues[self.state[self.DIALOG_POS]]
        dialogue = self.data_maps[dialog_index]
        n_turns = len(dialogue["dialogue"])
        return (self.state[self.TURN_POS]+1) >= n_turns

    def next_state(self):
        if self.is_done():
            self.state = self.reset() 
        else:
            self.state[self.TURN_POS] += 1
        return self.state
    
    def calculate_reward(self, new_transcript: str):  
        dialog_index = self.dialogues[self.state[self.DIALOG_POS]]
        dialogue = self.data_maps[dialog_index]
        turn_idx = self.state[self.TURN_POS]
        ori_transcript = dialogue["dialogue"][turn_idx]["transcript"]
        ori_gini = self.interface.gini_query(
            dialogue, turn_idx, ori_transcript)
        new_gini = self.interface.gini_query(
            dialogue, turn_idx, new_transcript)

        diff_gini = abs(new_gini-ori_gini)
        modification_rate = calculate_modif_rate(ori_transcript, new_transcript)
        beta = modification_rate if modification_rate<0.25 else -100
        reward = diff_gini/modification_rate + beta
        return reward

    def set_state(self, dialog_id : str):
        if dialog_id not in self.dialogues: 
            state = self.reset()
        else:
            state = [None] * self.STATE_ELEMENTS
            state[self.DIALOG_POS] = self.dialogues.index(dialog_id)
            state[self.TURN_POS] = 0
        self.state = state

    def step(self, action):
        actions, all_params = action 
        all_params = np.clip(
            all_params, a_min=np.zeros(self.action_parameter_size, dtype="float"), a_max=np.ones(self.action_parameter_size, dtype = "float"))
        action = (actions, all_params)
        turn_idx = int(self.state[self.TURN_POS])
        d_index = self.state[self.DIALOG_POS]
        dialog_index = self.dialogues[d_index]
        dialogue = self.data_maps[dialog_index]
        try :
            transcript = dialogue["dialogue"][turn_idx]["transcript"]
        except :
            print("turn_idx",  turn_idx)
            print("index", d_index)
            print("dialog index", dialog_index)
        new_transcript = self.compound_transfomer.apply(transcript, action)
        reward = self.calculate_reward(new_transcript)

        n_state = self.next_state()
        done = self.is_done()
        info = {}
        return n_state, reward, done, info

    def apply(self, action):
        actions, all_params = action
        all_params = np.clip(
            all_params, a_min=np.zeros(self.action_parameter_size, dtype="float"), a_max=np.ones(self.action_parameter_size, dtype = "float"))
        action = (actions, all_params)
        turn_idx = int(self.state[self.TURN_POS])
        d_index = self.state[self.DIALOG_POS]
        dialog_index = self.dialogues[d_index]
        dialogue = self.data_maps[dialog_index]
        transcript = dialogue["dialogue"][turn_idx]["transcript"]
        new_transcript = self.compound_transfomer.apply(transcript, action)
        dst = self.interface.dst_query(dialogue, turn_idx, new_transcript)
        return new_transcript, dst

if __name__ == '__main__':
    dialogue_simulator = DialogueSimulator()
    print(dialogue_simulator.action_parameter_size, "action parameter sizes")
    print(dialogue_simulator.n_hidden_action, "n hidden actions")


