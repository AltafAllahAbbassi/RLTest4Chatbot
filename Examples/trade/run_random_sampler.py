import argparse
from Examples.trade.interface4trade import TradeInterface
from RLTest4Chatbot.random_sampler import RandomSamplerTester
from Examples.trade.constants import TRADE_TEST_21, TRADE_TEST_22 


parser = argparse.ArgumentParser()
parser.add_argument('--model_interface', default = TradeInterface)
parser.add_argument('--data-file', default = TRADE_TEST_21, type=str) 
parser.add_argument('--save_dir', default="Examples/trade/Results/", type=str)
parser.add_argument('--top-k', default=1, type=int)
parser.add_argument("--rep", default=1, type=int)
args = parser.parse_args()
print(args.top_k)
random_sampler = RandomSamplerTester(save_dir=args.save_dir, data_file=args.data_file, top_k=args.top_k, interface=args.model_interface)
random_sampler.test_chatbot()

