import math
from gym import spaces
from RLTest4Chatbot.environments.environment import Environment
from RLTest4Chatbot.environments.utils.constants import STATE_ELEMENTS, DIALOG_POS, TURN_POS, MAX_WORDS, VALID_RATE, TRANSFORMATIONS, OBSERVATION_LOWER, OBSERVATION_UPPER
from RLTest4Chatbot.transformation.transformer import CompoundTransformer
from Examples.MultiWOZ.util import build_dict
import random
import numpy as np
import math

class DialogueSimulator(Environment):
    """
    Dialogue Simulator with parametric actions
    """
    def __init__(self, data_file, model_interface, hyrbid = True,  cumulative =  False):
        super().__init__()
        self.DIALOG_POS = DIALOG_POS
        self.TURN_POS = TURN_POS
        self.STATE_ELEMENTS =  STATE_ELEMENTS
        self.observation_shape = (self.STATE_ELEMENTS,)
        self.compound_transfomer = CompoundTransformer(TRANSFORMATIONS)
        self.ACTIONS = self.compound_transfomer.get_actions()
        self.hybrid = hyrbid
        if self.hybrid:
            self.cumulative = False
        else : 
            self.cumulative = cumulative
        self.interface = model_interface()
        self.data_file = data_file
        self.data_maps, _, self.dialogues = build_dict(self.data_file) 

        self.state, self.dialogue =  self.reset()
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
        self.state, self.dialogue = self.reset()
        
    def reset(self):
        state = [None] * STATE_ELEMENTS
        state[self.DIALOG_POS] = random.choice(
            list(range(len(self.dialogues))))
        state[self.TURN_POS] = 0
        dialog_index = state[self.DIALOG_POS]
        dialog_id = self.dialogues[dialog_index]
        dialogue = self.data_maps[dialog_id]
        return state, dialogue

    def render(self):
        pass

    def close(self):
        pass

    def is_done(self):
        # dialog_index = self.dialogues[self.state[self.DIALOG_POS]]
        # dialogue = self.data_maps[dialog_index]
        n_turns = len(self.dialogue["dialogue"])
        return (self.state[self.TURN_POS]+1) >= n_turns

    def next_state(self):
        if self.is_done():
            self.state, self.dialogue = self.reset() 
        else:
            self.state[self.TURN_POS] += 1
        return self.state
    
    def reward_func(self, ori_gini, new_gini, ori_transcript, new_transcript, trans_rate):
        diff_gini = math.exp(abs(new_gini-ori_gini))
        # modification_rate = calculate_modif_rate(ori_transcript, new_transcript)
        word_rate, char_rate = trans_rate
        beta = 0 if (word_rate<0.25 and char_rate <0.25 )else -1000
        all_trans_rate = math.sqrt(word_rate*word_rate + char_rate*char_rate)
        reward = diff_gini/all_trans_rate + beta if all_trans_rate else diff_gini
        return reward

    def calculate_reward(self, new_transcript, ori_transcript, trans_rate):  
        turn_idx = self.state[self.TURN_POS]
        ori_gini = self.interface.gini_query(
            self.dialogue, turn_idx, ori_transcript)
        new_gini = self.interface.gini_query(
            self.dialogue, turn_idx, new_transcript)
    
        reward = self.reward_func(ori_gini, new_gini, ori_transcript, new_transcript, trans_rate)
        return reward

    def set_state(self, dialog_id : str):
        if dialog_id not in self.dialogues: 
            state, dialogue = self.reset()
        else:
            state = [None] * self.STATE_ELEMENTS
            state[self.DIALOG_POS] = self.dialogues.index(dialog_id)
            state[self.TURN_POS] = 0
            dialogue = self.data_maps[dialog_id]
        self.state = state
        self.dialogue = dialogue

    def step(self, action):
        actions, all_params = action 
        all_params = np.clip(
            list(map(abs, all_params)), a_min=np.zeros(self.action_parameter_size, dtype="float"), a_max=np.ones(self.action_parameter_size, dtype = "float"))
        action = (actions, all_params)
        turn_idx = int(self.state[self.TURN_POS])
        transcript = self.dialogue["dialogue"][turn_idx]["transcript"]
        transcript = self.dialogue["dialogue"][turn_idx]["transcript"]
        new_transcript, trans_rate = self.compound_transfomer.apply(transcript, action)

        if self.hybrid:
            save = random.randint(0,1)
            if save :
                self.dialogue["dialogue"][turn_idx]["transcript"] = new_transcript
            
        elif self.cumulative :
            self.dialogue["dialogue"][turn_idx]["transcript"] = new_transcript
        reward = self.calculate_reward(new_transcript, transcript, trans_rate)

        done = self.is_done()
        n_state = self.next_state()
        info = {}
        return n_state, reward, done, info

    def apply(self, action):
        actions, all_params = action
        all_params = np.clip(
            list(map(abs, all_params)), a_min=np.zeros(self.action_parameter_size, dtype="float"), a_max=np.ones(self.action_parameter_size, dtype = "float"))
        action = (actions, all_params)
        turn_idx = int(self.state[self.TURN_POS])
        ori_transcript = self.dialogue["dialogue"][turn_idx]["transcript"]
        ori_gini = self.interface.gini_query(self.dialogue, turn_idx, ori_transcript)
        new_transcript, edit_rate = self.compound_transfomer.apply(ori_transcript, action)
        dst_gini = self.interface.dst_gini_query(self.dialogue, turn_idx, new_transcript)
        new_dst = dst_gini["Prediction"]
        joint_acc = dst_gini["Joint Acc"]
        new_gini = dst_gini["Gini"]
        reward = self.reward_func(ori_gini, new_gini, ori_transcript, new_transcript, edit_rate)
        done = self.is_done()
        n_state = self.next_state()
        info = {}
        return new_transcript, new_dst, reward, done, joint_acc,edit_rate, n_state, info

    
    def apply_trans(self, transcript, action):
        actions, all_params = action
        all_params = np.clip(
            list(map(abs, all_params)), a_min=np.zeros(self.action_parameter_size, dtype="float"), a_max=np.ones(self.action_parameter_size, dtype = "float"))
        action = (actions, all_params)
        new_transcript, trans_rate =self.compound_transfomer.apply(transcript, action)
        return new_transcript, trans_rate

