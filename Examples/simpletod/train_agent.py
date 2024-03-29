import argparse
from Examples.simpletod.interface4simpletod import SimpleTodInterface
from RLTest4Chatbot.agents.multi_pdqn import MultiPDQN
from RLTest4Chatbot.environments.dialogue_simulator import DialogueSimulator
from RLTest4Chatbot.train_tester import RL_Trainer
from Examples.simpletod.constants import S_TOD_TEST_21, S_TOD_TRAIN_21
from RLTest4Chatbot.evaluate_tester import ChatbotTester

# assert the train and test sets come from the same version
TEST_VES = S_TOD_TEST_21.split("/")[-1].split(".")[0][-2:]
TRAIN_VES = S_TOD_TRAIN_21.split("/")[-1].split(".")[0][-2:]
assert TEST_VES == TRAIN_VES

parser = argparse.ArgumentParser()
parser.add_argument('--top-k', default = 3, type=int)
parser.add_argument('--evaluation_episodes', default = 0, help='Episodes over which to evaluate after training.', type=int)
parser.add_argument('--episodes', default = 1, help='Number of epsiodes.', type=int) 
parser.add_argument('--save_dir', default="Examples/simpletod/Results/", help='Output directory.', type=str)
parser.add_argument('--interface', default = SimpleTodInterface)
parser.add_argument('--train-data-file', default = S_TOD_TRAIN_21, type=str) 
parser.add_argument('--test-data-file', default = S_TOD_TEST_21, type=str) 
parser.add_argument('--agent_name', default="Multi_PDQN", help="Prefix of output files", type=str)
parser.add_argument('--cumulative', default=False, type=bool)
parser.add_argument('--hybrid', default=True, type=bool)
parser.add_argument('--rep', default=1, type=int)
parser.add_argument('--save-freq', default=5, type=int)
parser.add_argument("--do-eval", default=True, type=bool)
parser.add_argument('--agent-name', default="Multi_PDQN", type=str)



     
args = parser.parse_args()
# top-k assertion
top_k = args.top_k
try : 
    assert top_k> 0 and top_k <=6 
    print("top_k", top_k)
except : 
    print("Top-k actions out of range")
    exit()
    
env = DialogueSimulator(args.train_data_file, args.interface, args.hybrid, args.cumulative)
agent = MultiPDQN(env.observation_space.spaces[0], env.action_space, args.top_k)
trainer = RL_Trainer(env, agent, args.save_dir, args.episodes, args.evaluation_episodes, args.top_k, args.agent_name,rep=args.rep, save_freq= args.save_freq)
trainer.train()

if args.do_eval and args.episodes == 1000:
    episodes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    env = DialogueSimulator(args.test_data_file, args.interface,args.hybrid,  args.cumulative)
    agent = MultiPDQN(env.observation_space.spaces[0], env.action_space, args.top_k)
    for p in episodes:
        chatbot_tester = ChatbotTester(env, agent, args.save_dir, args.top_k, args.test_data_file, args.agent_name, args.rep, p)
        chatbot_tester.test_chatbot()

