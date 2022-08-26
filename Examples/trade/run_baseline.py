import argparse
from Examples.trade.interface4trade import TradeInterface
from RLTest4Chatbot.baseline import DialTest
from Examples.trade.constants import TRADE_TEST_21, TRADE_TEST_22


parser = argparse.ArgumentParser()
parser.add_argument('--model_interface', default = TradeInterface)
parser.add_argument('--threashold', default = 0.01, type=float)
parser.add_argument('--data-file', default = TRADE_TEST_22, type=str) 
parser.add_argument('--max_trans', default=10, type=int)
parser.add_argument('--cumulative', default = False, type=bool)
parser.add_argument('--hybrid', default = True, type=bool)
parser.add_argument('--save_dir', default="Examples/trade/Results/baseline/", type=str)
parser.add_argument('--trans_name', default="synonym_replace", type=str)

 

     
args = parser.parse_args()
baseline = DialTest(args.model_interface, args.threashold,args.data_file, args.max_trans, args.cumulative, args.hybrid, args.save_dir, args.trans_name )
baseline.test_chatbot()
