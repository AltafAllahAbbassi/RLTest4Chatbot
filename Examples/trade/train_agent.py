import argparse
from Examples.trade.interface4trade import TradeInterface
from RLTest4Chatbot.agents.multi_pdqn import MultiPDQN
from RLTest4Chatbot.environments.dialogue_simulator import DialogueSimulator
from RLTest4Chatbot.train_tester import RL_Trainer
from Examples.trade.constants import TRADE_COR_DEV_MultiWoz21, TRADE_COR_DEV_MultiWoz22, TRADE_COR_TEST_MultiWoz21, TRADE_COR_TEST_MultiWoz22



parser = argparse.ArgumentParser()
parser.add_argument('--top-k', default = 2, type=int)
parser.add_argument('--evaluation_episodes', default=3, help='Episodes over which to evaluate after training.', type=int)
parser.add_argument('--episodes', default=3, help='Number of epsiodes.', type=int) 
parser.add_argument('--save_dir', default="Examples/trade/Results/", help='Output directory.', type=str)
parser.add_argument('--interface', default = TradeInterface)
parser.add_argument('--train-data-file', default = TRADE_COR_DEV_MultiWoz21, type=str) 
parser.add_argument('--test-data-file', default = TRADE_COR_TEST_MultiWoz21, type=str) 
parser.add_argument('--agent_name', default="Multi_PDQN", help="Prefix of output files", type=str)

     
args = parser.parse_args()

env = DialogueSimulator(args.train_data_file, args.interface)
agent = MultiPDQN(env.observation_space.spaces[0], env.action_space, args.top_k)
trainer = RL_Trainer(env, agent, args.save_dir, args.episodes, args.evaluation_episodes, args.top_k, args.agent_name)
trainer.train()
env.set_data_file(args.test_data_file)
trainer.evaluate()
