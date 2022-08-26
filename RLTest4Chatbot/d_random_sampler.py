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

class D_RandomSamplerTester: # dumb random sampler
    def __init__(self, save_dir , data_file, interface, max_trans, rep = 1, np_seed=0):
        self.save_dir = save_dir
        self.data_file = data_file
        self.rep = rep
        self.np_seed = np_seed
        np.random.seed(self.np_seed)
        self.save_file = self.data_file.split("/")[-1].split(".")[0] +  "_"  + str(self.rep) + ".json"
        self.data_maps, _, self.dialogues = build_dict(self.data_file) 
        self.interface = interface()
        self.max_trans = max_trans

        self.compound_transfomer = CompoundTransformer(TRANSFORMATIONS)
        self.ACTIONS = list(self.compound_transfomer.get_actions().values())
        self.acts = [ACT[0] for ACT in self.ACTIONS]
        self.num_actions = len(self.acts) 
        self.num_params = self.compound_transfomer.vector_size



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
                edit_rate = 0
                turn = dialogue_[turn_idx]
                transcript = turn["transcript"]
                transcript_ = transcript
                ground_truth = turn['belief_state']
                active = bool(random.getrandbits(1))
                if active : 
                    k = random.randint(1, self.num_actions)
                    acts = random.sample(self.acts, k)  
                    for act in acts : 
                        n_trans = random.randint(1, self.max_trans)
                        for _ in range(n_trans ):
                            if len(transcript_) != 0:
                                params = act.sample_one(transcript_)
                                transcript_, edit_rate_ = act.apply(transcript_, params)
                                edit_rate = edit_rate + edit_rate_
                dst_gini = self.interface.dst_gini_query(dialogue, turn_idx, transcript_)
                new_dst = dst_gini["Prediction"]
                joint_acc = dst_gini["Joint Acc"]
                to_save["ground_truth"] = ground_truth
                to_save["transcript"] = transcript
                to_save["turn_idx"] = turn_idx
                to_save["transcript_tran"] =transcript_
                to_save["transformation_rate"] = edit_rate
                to_save["joint_acc"] =joint_acc
                to_save["pred"] = new_dst
                to_save_dialog["dialogue"].append(to_save)
                terminal = turn_idx == len(dialogue_)-1
                turn_idx = turn_idx + 1
            exec_time = time.time() - start_t 
            to_save["exec_time"] =exec_time
            result.append(to_save_dialog)
        with open(os.path.join(self.save_dir, "DRandomSampler", self.save_file), "w") as f:
                json.dump(result, f, indent=10)

