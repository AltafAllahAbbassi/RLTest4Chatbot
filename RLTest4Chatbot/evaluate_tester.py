import os
import time
from tqdm import tqdm
import numpy as np
import json

class ChatbotTester:
    def __init__(self, environement, agent,  save_dir,top_k, data_file, title, rep, train_episodes):
        self.env = environement
        self.agent = agent 
        self.top_k = top_k
        self.data_file = data_file
        self.title = title
        self.rep = rep
        self.train_episodes = train_episodes
        self.save_file = self.data_file.split("/")[-1].split(".")[0] + "_" + self.title + "_" + str(self.top_k) + str(self.rep) + "_" + str(self.train_episodes) + ".json"
        self.save_dir = save_dir
        self.model_prefix = str(self.top_k) + "_" + self.title + "_" + str(self.rep) + "_" + str(self.train_episodes) + "_"
        self.data_maps, self.dialogues = self.env.data_maps, self.env.dialogues
        self.agent.load_models(os.path.join(self.save_dir, "Models", self.model_prefix))

    def test_chatbot(self):
        result = []
        for i  in tqdm(range(len(self.dialogues))):  
            start_t = time.time()
            index = self.dialogues[i] 
            dialogue = self.data_maps[index]
            to_save_dialog = {"dialog_id": dialogue["dialogue_idx"],
                              "dialogue": []
                             }
            self.env.set_state(index)
            total_reward = 0
            terminal = False
            while not terminal:
                to_save={
                }
                turn_idx = self.env.state[self.env.TURN_POS]
                try :
                    transcript = dialogue["dialogue"][turn_idx]["transcript"]
                except :
                    print(state)
                state = np.array(self.env.state, dtype=np.float32, copy=False)
                to_save["transcript"] = transcript
                action = self.agent.act(state)
                new_transcript, new_dst, reward, terminal,joint_acc,trans_rate, _, _ = self.env.apply(action)
                total_reward += reward
                d_actions, p_actions = action
                to_save["ground_truth"] = dialogue["dialogue"][turn_idx]["belief_state"]
                to_save["turn_idx"] = turn_idx
                to_save["pred"] = new_dst
                to_save["transcript_tran"] =new_transcript
                to_save["transformation_rate"] = trans_rate
                to_save["d_action"] = d_actions
                p_actions = list(p_actions)
                to_save["p_action"] = p_actions
                to_save["joint_acc"] =joint_acc
                to_save["reward"] = reward
                to_save_dialog["dialogue"].append(to_save)
            exec_time = time.time() - start_t   
            to_save_dialog["total_reward"] = total_reward
            to_save["exec_time"] =exec_time
            result.append(to_save_dialog)
            
        with open(os.path.join(self.save_dir, "Evaluation", self.save_file), "w") as f:
                json.dump(result, f, indent=10, cls=NpEncoder)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)