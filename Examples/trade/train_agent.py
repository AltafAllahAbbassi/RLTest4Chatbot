import argparse
from Examples.trade.interface4trade import TradeInterface
from RLTest4Chatbot.agents.multi_pdqn import MultiPDQN
from RLTest4Chatbot.environments.dialogue_simulator import DialogueSimulator
from RLTest4Chatbot.train_tester import RL_Trainer
from Examples.trade.constants import TRADE_TEST_21, TARDE_TRAIN_21


# assert the train and test sets come from the same version
TEST_VES = TRADE_TEST_21.split("/")[-1].split(".")[0][-2:]
TRAIN_VES = TARDE_TRAIN_21.split("/")[-1].split(".")[0][-2:]
assert TEST_VES == TRAIN_VES

parser = argparse.ArgumentParser()
parser.add_argument('--top-k', default = 2, type=int)
parser.add_argument('--evaluation_episodes', default=3, help='Episodes over which to evaluate after training.', type=int)
parser.add_argument('--episodes', default=2, help='Number of epsiodes.', type=int) 
parser.add_argument('--save_dir', default="Examples/trade/Results/", help='Output directory.', type=str)
parser.add_argument('--interface', default = TradeInterface)
parser.add_argument('--train-data-file', default = TARDE_TRAIN_21, type=str) 
parser.add_argument('--test-data-file', default = TRADE_TEST_21, type=str) 
parser.add_argument('--agent_name', default="Multi_PDQN", type=str)
parser.add_argument('--cumulative', default= True, type=bool)

     
args = parser.parse_args()

env = DialogueSimulator(args.train_data_file, args.interface, args.cumulative)
agent = MultiPDQN(env.observation_space.spaces[0], env.action_space, args.top_k)
trainer = RL_Trainer(env, agent, args.save_dir, args.episodes, args.evaluation_episodes, args.top_k, args.agent_name)
trainer.train()
env.set_data_file(args.test_data_file)
trainer.evaluate()
