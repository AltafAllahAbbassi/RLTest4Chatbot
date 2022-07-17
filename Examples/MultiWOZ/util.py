from RLTest4Chatbot.environments.utils.constants import TRANSFORMATIONS
from pyparsing import actions
from RLTest4Chatbot.transformation.transformer import CharInsert
import json
import random
import os
from xlwt import Workbook
from Examples.trade.trade_dst.utils.utils_multiWOZ_DST import Dataset
from Examples.MultiWOZ.constants import SLOT_MAPS
from RLTest4Chatbot.transformation.transformer import WordDrop, WordInsert, WordReplace, CharDrop, CharInsert, CharReplace
from RLTest4Chatbot.constants import MULTIWOZ_21_PATH
from RLTest4Chatbot.transformation.helpers import calculate_modif_rate

ACTIONS = [CharInsert(), CharDrop(), CharReplace(), WordInsert(), WordDrop(), WordReplace()]
TRANSFORMATIONS = ["Char Insert", "Char Drop", "Char Replace", "Word Insert", "Word Drop", "Word Replace"]

def build_dict(data_file): 
    """
    Inputs :data_file => a multiwoz datafile
    Outputs:data_maps => a dictionary where the keys are the dialog_ids and the values are the dialogs
            data => a list of all dialogs in the data file
            keys => all dialog_ids in the given data file
    """
    data_maps = {}
    with open(data_file) as f:
        data = json.load(f)
        for idx, dial in enumerate(data):
            data_maps[dial["dialogue_idx"]] = dial
            data_maps[str(idx)] = dial
            data_maps[idx] = dial
    all_keys = list(data_maps.keys())
    keys=[]
    for i in range(len(all_keys)):
        if i% 3 == 0 :
            keys.append(all_keys[i])
    data = list (list(data_maps[s_key] for s_key in keys))

    return data_maps, data, keys 

class MultiWozDataLoader(Dataset):
    def __init__(self, data_file: str, tokenizer):
        self.tokenizer = tokenizer
        self.data_file = data_file
        self.data = self.load_data()

    def load_data(self):
        """
        Gievn a data, it returns a list of all the dialogues turns, each is  under the format 
               {system    : system_transcript,
                user       : transcript,
                dialogue_id: dialogue_id,
                turn_id    : turn_id,
                domain_slot_value_map
                } 
        """
        with open(self.data_file) as f:
            data = []
            dials = json.load(f)
            for dial_dict in dials:
                for turn in dial_dict["dialogue"]:
                    domain_slot_value_maps = self.linear_turn_label(turn["turn_label"])
                    data_detail = {
                        "system":turn["system_transcript"], 
                        "user":turn["transcript"],
                        "domain_slot_value_maps": domain_slot_value_maps,
                        "dialogue_idx": dial_dict["dialogue_idx"],
                        "turn_idx": turn["turn_idx"]
                        }
                    data.append(data_detail)
        return data

    def linear_turn_label(self,turn_label: list):
        """
        Given a turn label, it returns a domain slot value map
        Input : turn label under the format [['slot1 value1'] ...]
        Output: domain slot value map under the format {'domain1' : {'slot1 : value1, ...}, ...}
        """
        domain_slot_value_maps = {}
        for (sub_domain,value) in turn_label:
            if(value=="none"):
                continue
            cur_domain,slot_name = sub_domain.split("-")

            if(slot_name in SLOT_MAPS):
                slot_name = SLOT_MAPS[slot_name]

            if(cur_domain not in domain_slot_value_maps):
                domain_slot_value_maps[cur_domain] = [[slot_name,value]]
            else:
                domain_slot_value_maps[cur_domain].append([slot_name,value])
        return domain_slot_value_maps

    def __len__(self):
        """
        Returns the length of the data file
        """
        return len(self.data)

    def get_dialID_turnID(self, idx: int):
        """
        Given the index in self.data returns the relative dialogue_idx and turn_idx
        """
        data_detail = self.data[idx]
        dialogue_idx = data_detail["dialogue_idx"].strip()
        turn_idx = int(data_detail["turn_idx"])
        return dialogue_idx, turn_idx

def generate_seed_turns(data_file, seed_size):
    """
    Returns an array of random turns 
    """
    _, data, _ = build_dict(data_file)
    seed_turns =[]

    for _ in range(seed_size):
        to_save = {}
        dialogue = random.choice(data)
        to_save["dialogue_idx"] = dialogue["dialogue_idx"]
        turns = dialogue["dialogue"]
        turn = random.choice(turns)
        to_save["turn_idx"] = turn["turn_idx"]
        to_save["transcript"] = turn["transcript"]
        seed_turns.append(to_save)
    return seed_turns

def generate_seed_dialogues(data_file, seed_size):
    """
    Returns an array of random turns 
    """

    seed_data = []
    _ , data, _ = build_dict(data_file)
    for _ in range(seed_size):
        seed_data.append(random.choice(data))

    return seed_data

def generate_consensus_file(data_file , save_file, size):
    """
    This file serves to determine the valid rates of each of our transformations
    """
    wb = Workbook()
    turns = generate_seed_turns(data_file, size)
    for i in range(len(ACTIONS)):
        print(TRANSFORMATIONS[i])
        action = ACTIONS[i]
        sheet = wb.add_sheet(TRANSFORMATIONS[i])
        col = 0
        row = 0
        sheet.write(row, col, "Transcript")
        sheet.write(row, col+1, "New Transcript")
        sheet.write(row, col+2, "Modification Rate")

        for j in range(len(turns)):
            row = row+1
            col = 0
            transcript = turns[j]["transcript"]
            params = action.sample(transcript)
            new_transcript = action.apply(transcript, params)
            modif_rate = calculate_modif_rate(transcript, new_transcript)
            sheet.write(row, col, transcript)
            sheet.write(row, col+1, new_transcript)
            sheet.write(row, col+2, modif_rate)
    wb.save('consensus.xls')

    


        # self.compound_transfomer = CompoundTransformer(TRANSFORMATIONS)
        # self.ACTIONS = self.compound_transfomer.get_actions()

def get_correct_data(data_file, model_interface, save_dir):
    _, data, _ = build_dict(data_file)
    correct_data = []
    for i in range(len(data)):
        dialogue = data[i]
        is_correct = True
        turn_idx = 0
        while ((turn_idx < len(dialogue['dialogue'])) and is_correct):
            transcript = dialogue["dialogue"][turn_idx]["transcript"]
            prediction = model_interface.dst_query(dialogue, turn_idx, transcript)
            if prediction['Joint Acc'] == 0.0:
                is_correct = False
            turn_idx = turn_idx+1
        if is_correct:
            correct_data.append(dialogue)

    if not os.path.exists(save_dir):
        exit()
    save_file = data_file.split("/")[-1]
    save_file = "corr_" + save_file
    with open(os.path.join(save_dir, save_file), "w") as f:
                json.dump(correct_data, f, indent=4)

def generate_predictions(data_file, model_interface, save_dir= None):
    # if save dir == none we dont save the file
    predictions = []
    _, data, _ = build_dict(data_file)
    for i in range(len(data)):
        dialogue = data[i]
        dialogue_idx = dialogue["dialogue_idx"]
        turns = dialogue["dialogue"]
        for t in range(len(turns)):
            turn = turns[t]
            turn_idx = turn["turn_idx"]
            transcript = turn["transcript"]
            turn_label = turn["turn_label"]
            prediction = model_interface.dst_query(dialogue, turn_idx, transcript)
            to_save = {
                "dialogue_idx" : dialogue_idx, 
                "turn_idx"  : turn_idx,
                "transcript" : transcript,
                "turn_label" : turn_label, 
                "pre_turn_label" : prediction["Prediction"],
                "joint_acc" : prediction["Joint Acc"]
            }
            predictions.append(to_save)
    if save_dir is None or not os.path.exists(save_dir):
        pass
    else : 
        save_file = data_file.split("/")[-1]
        save_file = "pred_" + save_file
        with open(os.path.join(save_dir, save_file), "w") as f:
            json.dump(predictions, f, indent=4)
    return predictions       

def get_prediction_stats(predictions):
    n_preds= len(predictions)
    joint_acc = 0
    for pred in predictions:
        joint_acc += pred["joint_acc"]
    return joint_acc/n_preds


def get_test_stats():
    pass
    


# print(generate_seed_turns("/home/altaf/Desktop/RLTest4chatbot/Examples/MultiWOZ/data/MultiWOZ_2.1/test_dials.json" , 3))
# generate_consensus_file("/home/altaf/Desktop/RLTest4chatbot/Examples/MultiWOZ/data/MultiWOZ_2.1/test_dials.json" , "save_file", 100)


