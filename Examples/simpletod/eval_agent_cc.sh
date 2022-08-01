#!/bin/bash
#SBATCH -t 2-12:00
#SBATCH --mem=5000

module load python/3.7
virtualenv --no-download ENV
source ENV/bin/activate

pip install --no-index -r requirements_cc.txt

pip install --no-index -e .


python Examples/simpletod/eval_agent.py \
 --top-k 3 \
 --train_episodes 2 \
 --save_dir Examples/simpletod/Results/ \
 --test-data-file Examples/simpletod/data/test_21.json  \
 --cumulative False   \
 --hybrid True  \
 --rep 1  \