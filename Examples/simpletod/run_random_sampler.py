import argparse
from Examples.simpletod.interface4simpletod import SimpleTodInterface
from RLTest4Chatbot.random_sampler import RandomSamplerTester
from Examples.simpletod.constants import S_TOD_TEST_21, S_TOD_TEST_22 


parser = argparse.ArgumentParser()
parser.add_argument('--model_interface', default = SimpleTodInterface)
parser.add_argument('--data-file', default = S_TOD_TEST_21, type=str) 
parser.add_argument('--save_dir', default="Examples/simpletod/Results/", type=str)
parser.add_argument('--top-k', default=4, type=int)
parser.add_argument("--rep", default=1, type=int)



args = parser.parse_args()
random_sampler = RandomSamplerTester(save_dir=args.save_dir, data_file=args.data_file, top_k=args.top_k, interface=args.model_interface)
random_sampler.test_chatbot()
