import argparse
from Examples.trade.interface4trade import TradeInterface
from RLTest4Chatbot.d_random_sampler import D_RandomSamplerTester
from Examples.trade.constants import TRADE_TEST_21, TRADE_TEST_22 


parser = argparse.ArgumentParser()
parser.add_argument('--model_interface', default = TradeInterface)
parser.add_argument('--data-file', default = TRADE_TEST_21, type=str) 
parser.add_argument('--save_dir', default="Examples/trade/Results/", type=str)
parser.add_argument('--max_trans', default= 40, type=int)
parser.add_argument("--rep", default=1, type=int)
args = parser.parse_args()
random_sampler = D_RandomSamplerTester(save_dir=args.save_dir, data_file=args.data_file, max_trans = args.max_trans,interface=args.model_interface)
random_sampler.test_chatbot()

