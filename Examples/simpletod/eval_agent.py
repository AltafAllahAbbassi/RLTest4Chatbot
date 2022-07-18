import argparse
from Examples.simpletod.interface4simpletod import  SimpleTodInterface
from RLTest4Chatbot.agents.multi_pdqn import MultiPDQN
from RLTest4Chatbot.environments.dialogue_simulator import DialogueSimulator
from RLTest4Chatbot.evaluate_tester import ChatbotTester
from Examples.simpletod.constants import S_TOD_TEST_21, S_TOD_TEST_22


parser = argparse.ArgumentParser()
parser.add_argument('--top-k', default = 2, type=int)
parser.add_argument('--save_dir', default="Examples/trade/Results/", type=str)
parser.add_argument('--interface', default = SimpleTodInterface)
parser.add_argument('--test-data-file', default = S_TOD_TEST_21, type=str) 
parser.add_argument('--agent_name', default="Multi_PDQN", type=str)

     
args = parser.parse_args()

env = DialogueSimulator(args.test_data_file, args.interface)
agent = MultiPDQN(env.observation_space.spaces[0], env.action_space, args.top_k)
chatbot_tester = ChatbotTester(env, agent, args.save_dir, args.top_k,args.test_data_file, args.agent_name)
chatbot_tester.test_chatbot()
