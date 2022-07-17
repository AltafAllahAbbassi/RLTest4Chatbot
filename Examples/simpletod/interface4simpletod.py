import torch 
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from Examples.simpletod.constants import S_TOD_MODEL_CHECKPOINT, S_TOD_EOB_TOKEN
from Examples.utils.interface import ModelInterface
from Examples.simpletod.helpers import dial2text, get_belief_new_dbsearch, convert_belief, format_bs, cal_join_acc
from RLTest4Chatbot.constants import ENVIRONMENT

class SimpleTodInterface(ModelInterface):

    def __init__(self):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(S_TOD_MODEL_CHECKPOINT)
        self.model = GPT2LMHeadModel.from_pretrained(S_TOD_MODEL_CHECKPOINT)
        self.break_tokens = self.tokenizer.encode(self.tokenizer._eos_token) \
            + self.tokenizer.encode('?') \
            + self.tokenizer.encode('!')
        self.cfg_n_ctx = self.model.config.n_ctx  # max length of the context encoding

    def dst_query(self, dialogue: dict, turn_id: int, user_input: str):
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
        self.model.eval()
        self.model.to(ENVIRONMENT)
        text = dial2text(dialogue, turn_id, user_input)
        indexed_tokens = self.tokenizer.encode(text)
        if (len(indexed_tokens) > self.cfg_n_ctx):
            indexed_tokens = indexed_tokens[-1 * self.MAX_LEN:]
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to(ENVIRONMENT)
        predicted_index = indexed_tokens[-1]
        with torch.no_grad():
            while predicted_index not in self.break_tokens:
                outputs = self.model(tokens_tensor)
                predictions = outputs[0]
                predicted_index = torch.argmax(predictions[0, -1, :]).item()
                indexed_tokens += [predicted_index]
                tokens_tensor = torch.tensor([indexed_tokens]).to(ENVIRONMENT)
                if len(indexed_tokens) > self.cfg_n_ctx:
                    break
                if self.tokenizer.decode(indexed_tokens).endswith(S_TOD_EOB_TOKEN):
                    break

        tmp_pred = self.tokenizer.decode(indexed_tokens)

        try:
            pred_belief_text = get_belief_new_dbsearch(tmp_pred)
            pred_beliefs = convert_belief(pred_belief_text)

            turn_target, turn_pred = format_bs(
                target=dialogue["dialogue"][turn_id]["belief_state"], pred=pred_beliefs)
            result = cal_join_acc(turn_pred=turn_pred, turn_target=turn_target)
            result["context"] = text
        except:
            return {"Prediction": [],
                    "Ground Truth": dialogue["dialogue"][turn_id]["belief_state"],
                    "Joint Acc": 0.0}

        return result

    def gini_query(self, dialogue: dict, turn_id: int, user_input: str):
        """
        Given a dialogue, turn_id, user_input, it calculates the gini value at that turn 
        Inputs :dialogue => a list of turns
                turn_id => the id of the turn, we are interessted about  
                user_input => the user transcript (response) at turn number = turn_id
        Output :gini value (= mean value of all gini values at every step )
        """
        self.model.eval()
        self.model.to(ENVIRONMENT)
        text = dial2text(dialogue, turn_id, user_input)
        indexed_tokens = self.tokenizer.encode(text)
        if (len(indexed_tokens) > self.cfg_n_ctx):
            indexed_tokens = indexed_tokens[-1 * self.cfg_n_ctx:]
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to(ENVIRONMENT)
        predicted_index = indexed_tokens[-1]
        ginis = []
        with torch.no_grad():
            while predicted_index not in self.break_tokens:
                outputs = self.model(tokens_tensor)
                predictions = outputs[0]
                predicted_softmax = nn.Softmax(dim=0)(predictions[0, -1, :])
                predicted_gini = 1 - \
                    torch.sum(torch.mul(predicted_softmax, predicted_softmax))
                ginis.append(predicted_gini.item())
                predicted_index = torch.argmax(predictions[0, -1, :]).item()
                indexed_tokens += [predicted_index]
                tokens_tensor = torch.tensor([indexed_tokens]).to(ENVIRONMENT)
                if len(indexed_tokens) > self.cfg_n_ctx:
                    break
                if self.tokenizer.decode(indexed_tokens).endswith(S_TOD_EOB_TOKEN):
                    break
        return sum(ginis)/len(ginis)

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
        self.model.eval()
        self.model.to(ENVIRONMENT)
        text = dial2text(dialogue, turn_id, user_input)
        indexed_tokens = self.tokenizer.encode(text)
        if (len(indexed_tokens) > self.cfg_n_ctx):
            indexed_tokens = indexed_tokens[-1 * self.MAX_LEN:]
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to(ENVIRONMENT)
        predicted_index = indexed_tokens[-1]
        ginis = []
        with torch.no_grad():
            while predicted_index not in self.break_tokens:
                outputs = self.model(tokens_tensor)
                predictions = outputs[0]
                predicted_softmax = nn.Softmax(dim=0)(predictions[0, -1, :])
                predicted_gini = 1 - \
                    torch.sum(torch.mul(predicted_softmax, predicted_softmax))
                ginis.append(predicted_gini.item())
                predicted_index = torch.argmax(predictions[0, -1, :]).item()
                indexed_tokens += [predicted_index]
                tokens_tensor = torch.tensor([indexed_tokens]).to(ENVIRONMENT)
                if len(indexed_tokens) > self.cfg_n_ctx:
                    break
                if self.tokenizer.decode(indexed_tokens).endswith(S_TOD_EOB_TOKEN):
                    break

        tmp_pred = self.tokenizer.decode(indexed_tokens)

        try:
            pred_belief_text = get_belief_new_dbsearch(tmp_pred)
            pred_beliefs = convert_belief(pred_belief_text)

            turn_target, turn_pred = format_bs(
                target=dialogue["dialogue"][turn_id]["belief_state"], pred=pred_beliefs)

            result = cal_join_acc(turn_pred=turn_pred, turn_target=turn_target)
            result["context"] = text
        except:
            result = {"Prediction": [],
                      "Ground Truth": dialogue["dialogue"][turn_id]["belief_state"],
                      "Joint Acc": 0.0}

        result["Gini"] = sum(ginis)/len(ginis)
        return result

if __name__ == '__main__':

    one_dialogue = {'dialogue_idx': 'PMUL3027.json', 'domains': ['train', 'attraction'], 'dialogue': [{'system_transcript': '', 'turn_idx': 0, 'belief_state': [{'slots': [['attraction-area', 'centre']], 'act': 'inform'}], 'turn_label': [['attraction-area', 'centre']], 'transcript': 'i am staying in the centre of town for the weekend, what is there to do there?', 'system_acts': [], 'domain': 'attraction'}, {'system_transcript': 'we have several things to do! architecture, colleges, museums...what type of attraction are you most interested in?', 'turn_idx': 1, 'belief_state': [{'slots': [['attraction-type', 'dontcare']], 'act': 'inform'}, {'slots': [['attraction-area', 'centre']], 'act': 'inform'}], 'turn_label': [['attraction-type', 'dontcare']], 'transcript': "it doesn't matter but can you recommend one and give me the entrance fee?", 'system_acts': ['type', ['type', 'architecture'], ['type', ' college'], ['type', ' museum'], ['choice', 'several']], 'domain': 'attraction'}, {'system_transcript': "i recommend castle galleries and it's free to get in!", 'turn_idx': 2, 'belief_state': [{'slots': [['attraction-type', 'dontcare']], 'act': 'inform'}, {'slots': [['attraction-area', 'centre']], 'act': 'inform'}, {'slots': [['train-departure', 'leicester']], 'act': 'inform'}], 'turn_label': [['train-departure', 'leicester']], 'transcript': "thanks! i'm also looking for a train that leaves leicester. ", 'system_acts': [], 'domain': 'train'}, {'system_transcript': 'i have plenty of trains departing from leicester, what destination did you have in mind?', 'turn_idx': 3, 'belief_state': [{'slots': [['attraction-type', 'dontcare']], 'act': 'inform'}, {'slots': [['attraction-area', 'centre']], 'act': 'inform'}, {'slots': [['train-destination', 'cambridge']], 'act': 'inform'}, {'slots': [['train-day', 'monday']], 'act': 'inform'}, {'slots': [['train-arriveby', '16:15']], 'act': 'inform'}, {'slots': [['train-departure', 'leicester']], 'act': 'inform'}], 'turn_label': [['train-destination', 'cambridge'], ['train-day', 'monday'], ['train-arriveby', '16:15']], 'transcript': "i'd like to go to cambridge.  i want to leave on monday and arrive by 16:15.", 'system_acts': [['choice', 'plenty'], ['depart', 'leicester'], 'dest'], 'domain': 'train'}, {'system_transcript': 'tr0330 departs at 14:09 and arrives by 15:54. would you like a ticket?', 'turn_idx': 4, 'belief_state': [{'slots': [['attraction-type', 'dontcare']], 'act': 'inform'}, {'slots': [['attraction-area', 'centre']], 'act': 'inform'}, {'slots': [['train-destination', 'cambridge']], 'act': 'inform'}, {'slots': [['train-day', 'monday']], 'act': 'inform'}, {'slots': [['train-arriveby', '16:15']], 'act': 'inform'}, {
        'slots': [['train-departure', 'leicester']], 'act': 'inform'}], 'turn_label': [], 'transcript': 'what is the total travel time?', 'system_acts': [['arrive', '15:54'], ['id', 'tr0330'], ['leave', '14:09']], 'domain': 'train'}, {'system_transcript': '105 minutes is the total travel time. can i help you with anything else?', 'turn_idx': 5, 'belief_state': [{'slots': [['attraction-type', 'dontcare']], 'act': 'inform'}, {'slots': [['attraction-area', 'centre']], 'act': 'inform'}, {'slots': [['train-destination', 'cambridge']], 'act': 'inform'}, {'slots': [['train-day', 'monday']], 'act': 'inform'}, {'slots': [['train-arriveby', '16:15']], 'act': 'inform'}, {'slots': [['train-departure', 'leicester']], 'act': 'inform'}], 'turn_label': [], 'transcript': 'would you be able to help me book this?', 'system_acts': [['time', '105 minutes ']], 'domain': 'train'}, {'system_transcript': 'yes, for how many tickets?', 'turn_idx': 6, 'belief_state': [{'slots': [['attraction-type', 'dontcare']], 'act': 'inform'}, {'slots': [['attraction-area', 'centre']], 'act': 'inform'}, {'slots': [['train-destination', 'cambridge']], 'act': 'inform'}, {'slots': [['train-day', 'monday']], 'act': 'inform'}, {'slots': [['train-arriveby', '16:15']], 'act': 'inform'}, {'slots': [['train-departure', 'leicester']], 'act': 'inform'}], 'turn_label': [], 'transcript': 'i would just like to find a train first, and get the info. i think i have the info i needed. ', 'system_acts': ['people'], 'domain': 'train'}, {'system_transcript': 'great. is there anything else that you need help with?', 'turn_idx': 7, 'belief_state': [{'slots': [['attraction-type', 'dontcare']], 'act': 'inform'}, {'slots': [['attraction-area', 'centre']], 'act': 'inform'}, {'slots': [['train-destination', 'cambridge']], 'act': 'inform'}, {'slots': [['train-day', 'monday']], 'act': 'inform'}, {'slots': [['train-arriveby', '16:15']], 'act': 'inform'}, {'slots': [['train-departure', 'leicester']], 'act': 'inform'}], 'turn_label': [], 'transcript': "no, i think that's all i need for now. thank you so much for your help!", 'system_acts': [], 'domain': 'train'}, {'system_transcript': 'you are welcome', 'turn_idx': 8, 'belief_state': [{'slots': [['attraction-type', 'dontcare']], 'act': 'inform'}, {'slots': [['attraction-area', 'centre']], 'act': 'inform'}, {'slots': [['train-destination', 'cambridge']], 'act': 'inform'}, {'slots': [['train-day', 'monday']], 'act': 'inform'}, {'slots': [['train-arriveby', '16:15']], 'act': 'inform'}, {'slots': [['train-departure', 'leicester']], 'act': 'inform'}], 'turn_label': [], 'transcript': 'you have been of great help', 'system_acts': [], 'domain': 'train'}]}

    trade_interface = SimpleTodInterface()
    print(trade_interface.dst_query(one_dialogue, 1,
          one_dialogue["dialogue"][1]["transcript"]))
    print(trade_interface.gini_query(one_dialogue, 1,
          one_dialogue["dialogue"][1]["transcript"]))
    print(trade_interface.dst_gini_query(one_dialogue,
          1, one_dialogue["dialogue"][1]["transcript"]))