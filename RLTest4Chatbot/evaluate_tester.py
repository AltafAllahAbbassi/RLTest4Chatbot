import argparse
import os
from tqdm import tqdm
import numpy as np
import json
from RLTest4Chatbot.environments.dialogue_simulator import DialogueSimulator
from RLTest4Chatbot.agents.multi_pdqn import MultiPDQN
from Examples.trade.interface4trade import TradeInterface

class ChatbotTester:
    def __init__(self, environement, agent,  save_dir,top_k, data_file, title):
        self.env = environement
        self.agent = agent 
        self.top_k = top_k
        self.data_file = data_file
        self.title = title
        self.save_file = self.data_file.split("/")[-1].split(".")[0] + "_" + self.title + "_" + str(self.top_k) + ".json"
        self.save_dir = save_dir
        self.model_prefix = str(self.top_k) + "_" + self.title + "_" 
        self.data_maps, self.dialogues = self.env.data_maps, self.env.dialogues
        self.agent.load_models(os.path.join(self.save_dir, "Models", self.model_prefix))

    def test_chatbot(self):
        result = []
        for i  in tqdm(range(len(self.dialogues))):        
            index = self.dialogues[i] 
            dialogue = self.data_maps[index]
            to_save_dialog = {"dialog_id": dialogue["dialogue_idx"],
                              "dialogue": []
                             }
            self.env.set_state(index)
            total_reward = 0
            terminal = False
            while not terminal:
                to_save={}
                turn_idx = self.env.state[self.env.TURN_POS]
                try :
                    transcript = dialogue["dialogue"][turn_idx]["transcript"]
                except :
                    print(state)
                state = np.array(self.env.state, dtype=np.float32, copy=False)
                to_save["transcript"] = transcript
                action = self.agent.act(state)
                new_transcript, new_dst = self.env.apply(action)
                state, reward, terminal, _ = self.env.step(action)
                total_reward += reward
                to_save["ground_truth"] = dialogue["dialogue"][turn_idx]["turn_label"]
                to_save["turn_idx"] = turn_idx
                to_save["pred"] = new_dst
                to_save["transcript_tran"] =new_transcript
                to_save_dialog["dialogue"].append(to_save)

            to_save_dialog["toral_reward"] = total_reward
            result.append(to_save_dialog)
        
        with open(os.path.join(self.save_dir, "Evaluation", self.save_file), "w") as f:
            json.dump(result, f, indent=4)
