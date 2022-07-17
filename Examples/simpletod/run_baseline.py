import argparse
from Examples.simpletod.interface4simpletod import SimpleTodInterface
from RLTest4Chatbot.baseline import DialTest
from RLTest4Chatbot.constants import MULTIWOZ_21_PATH, MULTIWOZ_22_PATH


parser = argparse.ArgumentParser()
parser.add_argument('--model_interface', default = SimpleTodInterface)
parser.add_argument('--threashold', default = 0.05, type=float)
parser.add_argument('--data-file', default = MULTIWOZ_21_PATH +"test_dials.json", type=str) 
parser.add_argument('--max_trans', default=10, type=int)
parser.add_argument('--cumulative', default = False, type=bool)
parser.add_argument('--save_dir', default="Examples/simpletod/Results/baseline/", type=str)
parser.add_argument('--trans_name', default="synonym_replace", type=str)

 

     
args = parser.parse_args()
baseline = DialTest(args.model_interface, args.threashold,args.data_file, args.max_trans, args.cumulative, args.save_dir, args.trans_name )
results = baseline.test_chatbot()
print("Sucess rate = ",  baseline.get_sucess_rate(results))