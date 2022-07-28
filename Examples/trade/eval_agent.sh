#!/bin/bash
#SBATCH -t 2-12:00
#SBATCH --mem=5000 
python Examples/trade/eval_agent.py \
 --top-k 3 \
 --train_episodes 2 \
 --save_dir Examples/trade/Results/ \
 --test-data-file Examples/trade/data/test_21.json  \
 --cumulative False   \
 --hybrid True  \
 --rep 1  \