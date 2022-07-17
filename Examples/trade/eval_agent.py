import argparse
from Examples.trade.interface4trade import TradeInterface
from RLTest4Chatbot.agents.multi_pdqn import MultiPDQN
from RLTest4Chatbot.environments.dialogue_simulator import DialogueSimulator
from RLTest4Chatbot.evaluate_tester import ChatbotTester
from Examples.trade.constants import TRADE_COR_TEST_MultiWoz22, TRADE_COR_TEST_MultiWoz21


parser = argparse.ArgumentParser()
parser.add_argument('--top-k', default = 2, type=int)
parser.add_argument('--save_dir', default="Examples/trade/Results/", help='Output directory.', type=str)
parser.add_argument('--interface', default = TradeInterface)
parser.add_argument('--test-data-file', default = TRADE_COR_TEST_MultiWoz22, type=str) 
parser.add_argument('--agent_name', default="Multi_PDQN", help="Prefix of output files", type=str)

     
args = parser.parse_args()

env = DialogueSimulator(args.test_data_file, args.interface)
agent = MultiPDQN(env.observation_space.spaces[0], env.action_space, args.top_k)
chatbot_tester = ChatbotTester(env, agent, args.save_dir, args.top_k,args.test_data_file, args.agent_name)
chatbot_tester.test_chatbot()
