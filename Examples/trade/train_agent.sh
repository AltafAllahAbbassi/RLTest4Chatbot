#!/bin/bash
#SBATCH -t 2-12:00
#SBATCH --mem=5000 
python Examples/trade/train_agent.py \
 --top-k 1 \
 --evaluation_episodes 0  \
 --episodes 3000 \
 --save_dir Examples/trade/Results/ \
 --train-data-file Examples/trade/data/train_21.json  \
 --test-data-file Examples/trade/data/test_21.json  \
 --cumulative False   \
 --hybrid True  \
 --rep 1  \
 --save-freq 500 \