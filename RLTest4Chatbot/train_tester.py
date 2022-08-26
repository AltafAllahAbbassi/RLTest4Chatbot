
import os
import time
import numpy as np
from tqdm import tqdm
import logging
logging.basicConfig(filename='output.log'  , level=logging.INFO ,format="%(message)s")



class RL_Trainer():
    def __init__(self, environement, agent, save_dir, episodes, eval_episodes, top_k, agent_name, seed = 0, display_freq = 5, rep = 1, save_freq = 500):
        self.env = environement
        self.agent = agent 
        self.top_k = top_k
        self.seed = seed
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        self.episodes = episodes
        self.eval_episodes = eval_episodes
        self.display_freq = display_freq
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.agent_name = agent_name
        self.rep = rep
        self.model_prefix =  str(self.top_k) + "_" + self.agent_name + "_" + str(self.rep) + "_" 
        
    def train(self):
        total_reward = 0.
        returns = []
        start_time = time.time()
        tqdm_bar = tqdm(range(self.episodes))
        for i in tqdm_bar:
            start_episode = time.time()
            if i % self.save_freq == 0 and i != 0:
                model_prefix = self.model_prefix + str(i) + "_"
                self.agent.save_models(os.path.join(self.save_dir, "Models", model_prefix))

            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32, copy=False)
            action = self.agent.act(state)
            episode_reward = 0.
            self.agent.start_episode()
            terminal = False
            
            if i % self.display_freq == 0:
                logging.info(f"#################################")
                logging.info(f"episode {i}")
            while not terminal:  

                    turn_idx = int(self.env.state[self.env.TURN_POS])
                    ori_transcript = self.env.dialogue["dialogue"][turn_idx]["transcript"]

                    ret = self.env.step(action)
                    next_state, reward, terminal, _ = ret
                    next_state = np.array(next_state, dtype=np.float32, copy=False)
                    next_action = self.agent.act(next_state)

                    new_transcript, trans_rate = self.env.apply_trans(ori_transcript, action)

                    ac_n, p_n = next_action
                    n_action_store = [ac_n[2*i+1] for i in range(len(ac_n)//2)]
                    n_action_store = [np.argmax(n_action_store)]
                    n_action_store.extend(p_n)

                    ac_, p_ = action
                    action_store = [ac_[2*i+1] for i in range(len(ac_)//2)]
                    action_store = [np.argmax(action_store)]
                    action_store.extend(p_)

                    self.agent.step(state, action_store, reward, next_state, n_action_store, terminal, i % self.display_freq and terminal)
                    state = next_state
                    episode_reward += reward
                    epsilon = self.agent.epsilon
                    logging.info(f"epsilon {epsilon}")
                    logging.info(f"transcript: {ori_transcript}")
                    logging.info(f"transcript transformed: {new_transcript}")
                    logging.info(f"edit rate{trans_rate}" )
                    logging.info(f"discrete actions {ac_n}")
                    logging.info(f"parametric actions {p_n}")
                    logging.info(f"state {state}")

                    if terminal:
                        break
            end_episode = time.time()
            self.agent.end_episode()
            returns.append(episode_reward)
            total_reward += episode_reward
            epi_time = end_episode-start_episode

            tqdm_bar.set_description("Episode: " + str(i) + " reward: " + str(round(episode_reward, 3)) +  " took " + str(round(epi_time,1)) + "seconds" )

            if i % self.display_freq == 0:
                print('{0:5s} R:{1:.4f} r:{2:.4f}'.format(str(i), total_reward / (i + 1), np.array(returns[-self.display_freq:]).mean()))
                epsilon = self.agent.epsilon
                logging.info(f"execution time {epi_time}")
                logging.info(f"##########################################")

        end_time = time.time()
        print("Took %.2f seconds" % (end_time - start_time))
        self.env.close()
        model_prefix = self.model_prefix +  str(self.episodes) + "_"
        self.agent.save_models(os.path.join(self.save_dir, "Models", model_prefix))
        print("Average return =", sum(returns) / len(returns))

    def evaluate(self): 
        model_prefix = self.model_prefix +  str(self.episodes) + "_"
        self.agent.load_models(os.path.join(self.save_dir, "Models", model_prefix))
        returns = []
        for i in tqdm(range(self.eval_episodes)):
            state, _ = self.env.reset()
            terminal = False
            total_reward = 0.
            while not terminal:
                state = np.array(state, dtype=np.float32, copy=False)
                action = self.agent.act(state)
                state, reward, terminal, _ = self.env.step(action)
                total_reward += reward
            returns.append(total_reward)
        mean_reward = round(np.array(returns).mean(), 4)
        print("Evaluation average reward = ", mean_reward)
        return np.array(returns).mean()
