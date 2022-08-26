import numpy as np 
from tqdm import tqdm
import time  
import random
import json
import os
from Examples.MultiWOZ.util import build_dict
from RLTest4Chatbot.environments.utils.constants import TRANSFORMATIONS
from RLTest4Chatbot.transformation.transformer import CompoundTransformer
from RLTest4Chatbot.environments.utils.constants import  TRANSFORMATIONS
from RLTest4Chatbot.agents.utils.utils import get_actions
from RLTest4Chatbot.evaluate_tester import NpEncoder


class RandomSamplerTester:
    def __init__(self, save_dir, top_k , data_file, interface,  rep = 1, np_seed=0):
        self.top_k = top_k
        self.save_dir = save_dir
        self.data_file = data_file
        self.rep = rep
        self.np_seed = np_seed
        np.random.seed(self.np_seed)
        self.save_file = self.data_file.split("/")[-1].split(".")[0] +  "_" + str(self.top_k) + str(self.rep) + ".json"
        self.data_maps, _, self.dialogues = build_dict(self.data_file) 
        self.compound_transfomer = CompoundTransformer(TRANSFORMATIONS)
        self.ACTIONS = self.compound_transfomer.get_actions()
        self.num_actions = len(self.ACTIONS) 
        self.num_params = self.compound_transfomer.vector_size
        self.interface = interface()

    def sample_discrete_softmax(self):
        random_digits = np.array([random.uniform(0,1) for i in range(self.num_actions)])
        random_softmax = np.array(random_digits/sum(random_digits))
        return random_softmax

    def sample_params(self):
        random_params = np.array([random.uniform(0,1) for i in range(self.num_params)])
        return random_params
    
    def test_chatbot(self):
        result = []
        for i  in tqdm(range(len(self.dialogues))):    

            start_t = time.time()
            turn_idx = 0
            terminal = False
            index = self.dialogues[i] 
            dialogue = self.data_maps[index]
            dialogue_ = dialogue["dialogue"]
            to_save_dialog = {"dialog_id": dialogue["dialogue_idx"],
                              "dialogue": []
                             }
            
            while not terminal:
                to_save={
                }
                d_action = self.sample_discrete_softmax()
                d_action = get_actions(list(d_action), self.top_k)
                p_action = self.sample_params()
                action = (d_action, p_action)
                turn = dialogue_[turn_idx]
                transcript = turn["transcript"]
                new_transcript, trans_rate = self.compound_transfomer.apply(transcript,action )
                ground_truth = turn['belief_state']
                dst_gini = self.interface.dst_gini_query(dialogue, turn_idx, new_transcript)
                new_dst = dst_gini["Prediction"]
                joint_acc = dst_gini["Joint Acc"]
                new_gini = dst_gini["Gini"]
                to_save["ground_truth"] = ground_truth
                to_save["transcript"] = transcript
                to_save["turn_idx"] = turn_idx
                to_save["transcript_tran"] =new_transcript
                to_save["transformation_rate"] = trans_rate
                to_save["d_action"] = d_action
                p_action = list(p_action)
                to_save["p_action"] = p_action
                to_save["joint_acc"] =joint_acc
                to_save["pred"] = new_dst
                to_save_dialog["dialogue"].append(to_save)
                terminal = turn_idx == len(dialogue_)-1
                turn_idx = turn_idx + 1
            exec_time = time.time() - start_t 
            to_save["exec_time"] =exec_time
            result.append(to_save_dialog)

        with open(os.path.join(self.save_dir, "RandomSampler", self.save_file), "w") as f:
                json.dump(result, f, indent=10, cls=NpEncoder)
