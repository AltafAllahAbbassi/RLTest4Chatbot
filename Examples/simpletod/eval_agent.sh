#!/bin/bash
#SBATCH -t 2-12:00
#SBATCH --mem=5000 
python Examples/simpletod/eval_agent.py \
 --top-k 3 \
 --train_episodes 1000 \
 --save_dir Examples/simpletod/Results/ \
 --test-data-file Examples/simpletod/data/test_21.json  \
 --cumulative False   \
 --hybrid True  \
 --rep 1  \