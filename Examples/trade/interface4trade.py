import json
import pickle
import torch
import sys
from Examples.trade.trade_dst.utils.utils_multiWOZ_DST import get_slot_information, Lang, get_seq, read_GUI_langs
from Examples.trade.trade_dst.models.TRADE import TRADE
from Examples.utils.interface import ModelInterface
from Examples.trade.constants import TRADE_MODEL_CHECKPOINT, TRADE_PATH, TRADE_ONTOLOGY
sys.path.append(TRADE_PATH) #need this because of pickle.load

class TradeInterface(ModelInterface):
    def __init__(self):
        ontology = json.load(
            open(TRADE_ONTOLOGY, 'r'))
        self.SLOT_LIST = get_slot_information(ontology)
        self.gating_dict = {"ptr": 0, "dontcare": 1, "none": 2}
        self.lang, self.mem_lang = Lang(), Lang()
        self.lang.index_words(self.SLOT_LIST, 'slot')
        self.mem_lang.index_words(self.SLOT_LIST, 'slot')
        lang_name = 'lang-all.pkl'
        mem_lang_name = 'mem-lang-all.pkl'
        folder_name = TRADE_PATH + "baseline/"
        with open(folder_name+lang_name, 'rb') as handle:
            self.lang = pickle.load(handle)
        with open(folder_name+mem_lang_name, 'rb') as handle:
            self.mem_lang = pickle.load(handle)
        self.lang_ = [self.lang, self.mem_lang]
        self.HDD = TRADE_MODEL_CHECKPOINT.split(
            '/')[-1].split("HDD")[1].split("BSZ")[0]  # hidden size of the trained model
        self.model = TRADE(
            int(self.HDD),
            # lang=lang,
            path=TRADE_MODEL_CHECKPOINT,
            task='dst',
            lr=0,
            lang=self.lang_,
            dropout=0,
            # Trade model accepts slots as a list of all slots, train slots, dev slots, and test slots
            slots=[self.SLOT_LIST]*4,
            gating_dict=self.gating_dict)
        self.model.encoder.train(False)
        self.model.decoder.train(False)

    def gini_query(self, dialogue: dict, turn_id: int, user_input: str):
        """
        Given a dialogue, turn_id, user_input, it calculates the gini value at that turn 
        Inputs :dialogue => a list of turns
                turn_id => the id of the turn, we are interessted about  
                user_input => the user transcript (response) at turn number = turn_id
        Output :gini value (= mean value of all gini values for each slot )
        """
        ginis = []
        pair_test, _, _ = read_GUI_langs(
            dialogue, turn_id, user_input, self.gating_dict, self.SLOT_LIST)
        test = get_seq(pair_test, self.lang, self.mem_lang)
        data_ = []
        for j, data in enumerate(test):
            data_.append(data)
        assert(len(data_) == 1)
        data_ = data_[0]
        word_points, _, _, _ = self.model.encode_and_decode(
            data, False, self.SLOT_LIST)
        for slot in word_points:
            gini = []
            for r in slot[0]:
                gini.append((1-torch.sum(torch.mul(r, r)).item()))
            ginis.append(sum(gini)/len(gini))
        return sum(ginis)/len(ginis)

    def dst_query(self, dialogue: dict, turn_id, user_input):
        """
        Given a dialogue, turn_id, user_input, it predicts the belief state at that turn 
        Inputs :dialogue => a list of turns
                turn_id => the id of the turn, we are interessted about  
                user_input => the user transcript (response) at turn number = turn_id
        Output :{Prediction : belief state predicted,
                Ground Truth : belief state,
                Joint Acc : 0 or 1
            }
        """
        pair_test, _, _ = read_GUI_langs(
            dialogue, turn_id, user_input, self.gating_dict, self.SLOT_LIST)
        test = get_seq(pair_test, self.lang, self.mem_lang)
        result = self.model.query(test, 1e7, self.SLOT_LIST, True)
        result.update({"context": test.dataset[0]["context_plain"]})
        result["Ground Truth"] = sorted(result["Ground Truth"])
        result["Prediction"] = sorted(result["Prediction"])
        return result

    def dst_gini_query(self, dialogue: dict, turn_id: int, user_input: str):
        """
        Given a dialogue, turn_id, user_input, it predicts the belief state and calculates the gini at that turn 
        Inputs :dialogue => a list of turns
                turn_id => the id of the turn, we are interessted about  
                user_input => the user transcript (response) at turn number = turn_id
        Output :{Prediction : belief state predicted,
                Ground Truth : belief state,
                Joint Acc : 0 or 1,
                Gini : [0:1]  (= mean value of all gini values at every step )
            }
        """
        pair_test, _, _ = read_GUI_langs(
            dialogue, turn_id, user_input, self.gating_dict, self.SLOT_LIST)
        test = get_seq(pair_test, self.lang, self.mem_lang)
        result = self.model.query(test, 1e7, self.SLOT_LIST, True)
        result.update({"context": test.dataset[0]["context_plain"]})
        result["Ground Truth"] = sorted(result["Ground Truth"])
        result["Prediction"] = sorted(result["Prediction"])

        data_ = []
        ginis = []
        for j, data in enumerate(test):
            data_.append(data)

        assert(len(data_) == 1)
        data_ = data_[0]
        word_points, _, _, _ = self.model.encode_and_decode(
            data, False, self.SLOT_LIST)
        for slot in word_points:
            gini = []
            for r in slot[0]:
                gini.append((1-torch.sum(torch.mul(r, r)).item()))
            ginis.append(sum(gini)/len(gini))

        result["Gini"] = sum(ginis)/len(ginis)
        return result

if __name__ == '__main__':

    one_dialogue = {'dialogue_idx': 'PMUL3027.json', 'domains': ['train', 'attraction'], 'dialogue': [{'system_transcript': '', 'turn_idx': 0, 'belief_state': [{'slots': [['attraction-area', 'centre']], 'act': 'inform'}], 'turn_label': [['attraction-area', 'centre']], 'transcript': 'i am staying in the centre of town for the weekend, what is there to do there?', 'system_acts': [], 'domain': 'attraction'}, {'system_transcript': 'we have several things to do! architecture, colleges, museums...what type of attraction are you most interested in?', 'turn_idx': 1, 'belief_state': [{'slots': [['attraction-type', 'dontcare']], 'act': 'inform'}, {'slots': [['attraction-area', 'centre']], 'act': 'inform'}], 'turn_label': [['attraction-type', 'dontcare']], 'transcript': "it doesn't matter but can you recommend one and give me the entrance fee?", 'system_acts': ['type', ['type', 'architecture'], ['type', ' college'], ['type', ' museum'], ['choice', 'several']], 'domain': 'attraction'}, {'system_transcript': "i recommend castle galleries and it's free to get in!", 'turn_idx': 2, 'belief_state': [{'slots': [['attraction-type', 'dontcare']], 'act': 'inform'}, {'slots': [['attraction-area', 'centre']], 'act': 'inform'}, {'slots': [['train-departure', 'leicester']], 'act': 'inform'}], 'turn_label': [['train-departure', 'leicester']], 'transcript': "thanks! i'm also looking for a train that leaves leicester. ", 'system_acts': [], 'domain': 'train'}, {'system_transcript': 'i have plenty of trains departing from leicester, what destination did you have in mind?', 'turn_idx': 3, 'belief_state': [{'slots': [['attraction-type', 'dontcare']], 'act': 'inform'}, {'slots': [['attraction-area', 'centre']], 'act': 'inform'}, {'slots': [['train-destination', 'cambridge']], 'act': 'inform'}, {'slots': [['train-day', 'monday']], 'act': 'inform'}, {'slots': [['train-arriveby', '16:15']], 'act': 'inform'}, {'slots': [['train-departure', 'leicester']], 'act': 'inform'}], 'turn_label': [['train-destination', 'cambridge'], ['train-day', 'monday'], ['train-arriveby', '16:15']], 'transcript': "i'd like to go to cambridge.  i want to leave on monday and arrive by 16:15.", 'system_acts': [['choice', 'plenty'], ['depart', 'leicester'], 'dest'], 'domain': 'train'}, {'system_transcript': 'tr0330 departs at 14:09 and arrives by 15:54. would you like a ticket?', 'turn_idx': 4, 'belief_state': [{'slots': [['attraction-type', 'dontcare']], 'act': 'inform'}, {'slots': [['attraction-area', 'centre']], 'act': 'inform'}, {'slots': [['train-destination', 'cambridge']], 'act': 'inform'}, {'slots': [['train-day', 'monday']], 'act': 'inform'}, {'slots': [['train-arriveby', '16:15']], 'act': 'inform'}, {
        'slots': [['train-departure', 'leicester']], 'act': 'inform'}], 'turn_label': [], 'transcript': 'what is the total travel time?', 'system_acts': [['arrive', '15:54'], ['id', 'tr0330'], ['leave', '14:09']], 'domain': 'train'}, {'system_transcript': '105 minutes is the total travel time. can i help you with anything else?', 'turn_idx': 5, 'belief_state': [{'slots': [['attraction-type', 'dontcare']], 'act': 'inform'}, {'slots': [['attraction-area', 'centre']], 'act': 'inform'}, {'slots': [['train-destination', 'cambridge']], 'act': 'inform'}, {'slots': [['train-day', 'monday']], 'act': 'inform'}, {'slots': [['train-arriveby', '16:15']], 'act': 'inform'}, {'slots': [['train-departure', 'leicester']], 'act': 'inform'}], 'turn_label': [], 'transcript': 'would you be able to help me book this?', 'system_acts': [['time', '105 minutes ']], 'domain': 'train'}, {'system_transcript': 'yes, for how many tickets?', 'turn_idx': 6, 'belief_state': [{'slots': [['attraction-type', 'dontcare']], 'act': 'inform'}, {'slots': [['attraction-area', 'centre']], 'act': 'inform'}, {'slots': [['train-destination', 'cambridge']], 'act': 'inform'}, {'slots': [['train-day', 'monday']], 'act': 'inform'}, {'slots': [['train-arriveby', '16:15']], 'act': 'inform'}, {'slots': [['train-departure', 'leicester']], 'act': 'inform'}], 'turn_label': [], 'transcript': 'i would just like to find a train first, and get the info. i think i have the info i needed. ', 'system_acts': ['people'], 'domain': 'train'}, {'system_transcript': 'great. is there anything else that you need help with?', 'turn_idx': 7, 'belief_state': [{'slots': [['attraction-type', 'dontcare']], 'act': 'inform'}, {'slots': [['attraction-area', 'centre']], 'act': 'inform'}, {'slots': [['train-destination', 'cambridge']], 'act': 'inform'}, {'slots': [['train-day', 'monday']], 'act': 'inform'}, {'slots': [['train-arriveby', '16:15']], 'act': 'inform'}, {'slots': [['train-departure', 'leicester']], 'act': 'inform'}], 'turn_label': [], 'transcript': "no, i think that's all i need for now. thank you so much for your help!", 'system_acts': [], 'domain': 'train'}, {'system_transcript': 'you are welcome', 'turn_idx': 8, 'belief_state': [{'slots': [['attraction-type', 'dontcare']], 'act': 'inform'}, {'slots': [['attraction-area', 'centre']], 'act': 'inform'}, {'slots': [['train-destination', 'cambridge']], 'act': 'inform'}, {'slots': [['train-day', 'monday']], 'act': 'inform'}, {'slots': [['train-arriveby', '16:15']], 'act': 'inform'}, {'slots': [['train-departure', 'leicester']], 'act': 'inform'}], 'turn_label': [], 'transcript': 'you have been of great help', 'system_acts': [], 'domain': 'train'}]}

    trade_interface = TradeInterface()
    print(trade_interface.dst_query(one_dialogue, 1,
          one_dialogue["dialogue"][1]["transcript"]))
    print(trade_interface.gini_query(one_dialogue, 1,
          one_dialogue["dialogue"][1]["transcript"]))
    print(trade_interface.dst_gini_query(one_dialogue,
          1, one_dialogue["dialogue"][1]["transcript"]))