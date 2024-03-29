import argparse
from Examples.trade.interface4trade import TradeInterface
from RLTest4Chatbot.agents.multi_pdqn import MultiPDQN
from RLTest4Chatbot.environments.dialogue_simulator import DialogueSimulator
from RLTest4Chatbot.evaluate_tester import ChatbotTester
from Examples.trade.constants import TRADE_TEST_21, TRADE_TEST_22
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--top-k', default = 2, type=int)
parser.add_argument('--save_dir', default="Examples/trade/Results/", help='Output directory.', type=str)
parser.add_argument('--interface', default = TradeInterface)
parser.add_argument('--test-data-file', default = TRADE_TEST_21, type=str) 
parser.add_argument('--agent_name', default="Multi_PDQN", type=str)
parser.add_argument('--cumulative', default= False, type=bool)
parser.add_argument('--hybrid', default= True, type=bool)
parser.add_argument('--rep', default= 10, type=int)
parser.add_argument('--train_episodes', default= 1000, type=int)


args = parser.parse_args()
env = DialogueSimulator(args.test_data_file, args.interface,args.hybrid,  args.cumulative)
agent = MultiPDQN(env.observation_space.spaces[0], env.action_space, args.top_k)
chatbot_tester = ChatbotTester(env, agent, args.save_dir, args.top_k,args.test_data_file, args.agent_name, args.rep, args.train_episodes)
chatbot_tester.test_chatbot()