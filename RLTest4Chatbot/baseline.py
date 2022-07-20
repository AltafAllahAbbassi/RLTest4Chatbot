from tqdm import tqdm
import copy
import time
import os
import json
from  Examples.MultiWOZ.util import build_dict
from RLTest4Chatbot.transformation.transformer import apply, WwmInsert, BackTranslation, SynonymReplace


TRANSFROMATIONS = {
 "wwinsert": WwmInsert(),
 "back_translation" : BackTranslation(),
 "synonym_replace" :  SynonymReplace()
}
   
class DialTest():
    def __init__(self, model_interface, threashold, data_file, max_trans, cumulative, save_dir, trans_name):
        self.model_interface = model_interface()
        self.threashold = threashold
        self.data_file = data_file
        self.save_file = data_file.split("/")[-1]
        self.transfromations = TRANSFROMATIONS
        self.trans_name = trans_name
        self.max_trans = max_trans
        self.cumulative = cumulative
        self.data_maps, self.data, self.keys = build_dict(self.data_file)
        self.save_dir = save_dir
        
    def set_transformation(self, trans_name):
        self.trans_name = trans_name

    def test_chatbot(self):
        results = []
        save_file = self.trans_name + "_" + str(self.cumulative) + "_" + self.save_file
        for i in tqdm(range(len(self.data))):
            dialogue = self.data[i]  #backup clean dialogue
            dialogue_ = copy.deepcopy(dialogue)
            dialogue_idx =  dialogue["dialogue_idx"]
            turns = dialogue["dialogue"]
            for t in range(len(turns)):
                to_save= {}
                start = time.time()
                turn = turns[t]
                turn_idx = turn["turn_idx"]
                ori_transcript = turn["transcript"]
                new_transcript =  ori_transcript
                ori_gini = self.model_interface.gini_query(dialogue_, turn_idx, ori_transcript)
                new_gini = ori_gini
                n_trans = 0
                while(
                    (abs(new_gini-ori_gini) < self.threashold) and
                    n_trans < self.max_trans
                    ):
                    new_transcript = apply(new_transcript, self.trans_name, self.transfromations)
                    dst_gini = self.model_interface.dst_gini_query(dialogue_, turn_idx, new_transcript)
                    new_gini = dst_gini["Gini"]
                    n_trans = n_trans + 1
                if self.cumulative:
                    dialogue_["dialogue"][turn_idx]["transcript"] = new_transcript
                execution_time = time.time() - start
                to_save["dialogue_idx"] = dialogue_idx
                to_save["turn_idx"] = turn_idx
                to_save["ori_transcript"] = ori_transcript
                to_save["new_transcript"] = new_transcript
                to_save["n_trans"] = n_trans
                to_save["exec_time"] = execution_time
                to_save["joint_acc"] =dst_gini['Joint Acc']
                results.append(to_save)

        with open(os.path.join(self.save_dir, save_file), "w") as f:
                json.dump(results, f, indent=4)
        
        return results
    
    def get_sucess_rate(self, results):
        n_res= len(results)
        joint_acc = 0
        for res in results:
            joint_acc += res["joint_acc"]
        return joint_acc/n_res

        