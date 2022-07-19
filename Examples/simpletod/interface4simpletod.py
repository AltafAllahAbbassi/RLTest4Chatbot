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
