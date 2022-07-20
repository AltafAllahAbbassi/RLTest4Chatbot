import argparse
from Examples.simpletod.interface4simpletod import SimpleTodInterface
from RLTest4Chatbot.agents.multi_pdqn import MultiPDQN
from RLTest4Chatbot.environments.dialogue_simulator import DialogueSimulator
from RLTest4Chatbot.train_tester import RL_Trainer
from Examples.simpletod.constants import S_TOD_TEST_21, S_TOD_TRAIN_21

# assert the train and test sets come from the same version
TEST_VES = S_TOD_TEST_21.split("/")[-1].split(".")[0][-2:]
TRAIN_VES = S_TOD_TRAIN_21.split("/")[-1].split(".")[0][-2:]
assert TEST_VES == TRAIN_VES

parser = argparse.ArgumentParser()
parser.add_argument('--top-k', default = 3, type=int)
parser.add_argument('--evaluation_episodes', default = 10, help='Episodes over which to evaluate after training.', type=int)
parser.add_argument('--episodes', default = 100, help='Number of epsiodes.', type=int) 
parser.add_argument('--save_dir', default="Examples/simpletod/Results/", help='Output directory.', type=str)
parser.add_argument('--interface', default = SimpleTodInterface)
parser.add_argument('--train-data-file', default = S_TOD_TRAIN_21, type=str) 
parser.add_argument('--test-data-file', default = S_TOD_TEST_21, type=str) 
parser.add_argument('--agent_name', default="Multi_PDQN", help="Prefix of output files", type=str)
parser.add_argument('--cumulative', default=False, type=bool)

     
args = parser.parse_args()

env = DialogueSimulator(args.train_data_file, args.interface, args.cumulative)
agent = MultiPDQN(env.observation_space.spaces[0], env.action_space, args.top_k)
trainer = RL_Trainer(env, agent, args.save_dir, args.episodes, args.evaluation_episodes, args.top_k, args.agent_name)
trainer.train()
env.set_data_file(args.test_data_file)
trainer.evaluate()