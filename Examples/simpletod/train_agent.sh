#!/bin/bash
#SBATCH -t 2-12:00
#SBATCH --mem=5000 
python Examples/simpletod/train_agent.py \
 --top-k 3 \
 --evaluation_episodes 0  \
 --episodes 1000 \
 --save_dir Examples/simpletod/Results/ \
 --train-data-file Examples/simpletod/data/train_21.json  \
 --test-data-file Examples/simpletod/data/test_21.json  \
 --cumulative False   \
 --hybrid True  \
 --rep 1  \
  --save_freq 5 \