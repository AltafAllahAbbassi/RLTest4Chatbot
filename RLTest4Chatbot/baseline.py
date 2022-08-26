from tqdm import tqdm
import copy
import time
import os
import json
import random
from  Examples.MultiWOZ.util import build_dict
from RLTest4Chatbot.transformation.transformer import apply, WwmInsert, BackTranslation, SynonymReplace
from RLTest4Chatbot.transformation.helpers import jaro_distance

TRANSFROMATIONS = {
 "wwinsert": WwmInsert(),
 "back_translation" : BackTranslation(),
 "synonym_replace" :  SynonymReplace()
}
   
class DialTest():
    def __init__(self, model_interface, threashold, data_file, max_trans, cumulative, hybrid, save_dir, trans_name, rep = 1):
        self.model_interface = model_interface()
        self.threashold = threashold
        self.data_file = data_file
        self.rep = rep
        self.transfromations = TRANSFROMATIONS
        self.trans_name = trans_name
        self.max_trans = max_trans
        self.hybrid = hybrid
        if self.hybrid : 
            self.cumulative = False
        else : 
            self.cumulative = cumulative
        self.cumulative = cumulative
        self.data_maps, self.data, self.keys = build_dict(self.data_file)
        self.save_dir = save_dir
        self.save_file = data_file.split("/")[-1].split(".")[0] +  "_" + self.trans_name +"_" + str(self.threashold)+"_"+str(self.rep) + ".json"

        
    def set_transformation(self, trans_name):
        self.trans_name = trans_name

    def test_chatbot(self):
        results = []
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
                edit_rate = 0
                while(
                    (abs(new_gini-ori_gini) < self.threashold) and
                    n_trans < self.max_trans
                    ):
                    new_transcript, edit_rate_ = apply(new_transcript, self.trans_name, self.transfromations)
                    dst_gini = self.model_interface.dst_gini_query(dialogue_, turn_idx, new_transcript)
                    new_gini = dst_gini["Gini"]
                    n_trans = n_trans + 1
                    edit_rate = edit_rate + edit_rate_
                
                if self.hybrid : 
                    save = random.randint(0,1)
                    if save :
                        dialogue_["dialogue"][turn_idx]["transcript"] = new_transcript

                elif self.cumulative:
                    dialogue_["dialogue"][turn_idx]["transcript"] = new_transcript
                execution_time = time.time() - start
                edit_rate_ = jaro_distance(ori_transcript, new_transcript)
                to_save["transformation_rate"] = edit_rate_
                to_save["dialog_id"] = dialogue_idx
                to_save["turn_idx"] = turn_idx
                to_save["transcript"] = ori_transcript
                to_save["transcript_tran"] = new_transcript
                to_save["n_trans"] = n_trans
                to_save["exec_time"] = execution_time
                to_save["joint_acc"] =dst_gini['Joint Acc']
                results.append(to_save)

        with open(os.path.join(self.save_dir, self.save_file), "w") as f:
                json.dump(results, f, indent=4)
        
        return results
    
    def get_sucess_rate(self, results):
        n_res= len(results)
        joint_acc = 0
        for res in results:
            joint_acc += res["joint_acc"]
        return joint_acc/n_res


        