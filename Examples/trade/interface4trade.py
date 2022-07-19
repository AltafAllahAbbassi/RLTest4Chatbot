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

