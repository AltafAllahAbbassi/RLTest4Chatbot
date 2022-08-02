#!/bin/bash
#SBATCH -t 2-12:00
#SBATCH --mem=5000
#SBATCH --cpus-per-task=24 


module load python/3.7
virtualenv --no-download ENV
source ENV/bin/activate

pip install --no-index -r requirements_cc.txt

pip install --no-index -e .

python Examples/simpletod/train_agent.py \
 --top-k 3 \
 --evaluation_episodes 0  \
 --episodes 10 \
 --save_dir Examples/simpletod/Results/ \
 --train-data-file Examples/simpletod/data/train_21.json  \
 --test-data-file Examples/simpletod/data/test_21.json  \
 --cumulative False   \
 --hybrid True  \
 --rep 1  \
 --save_freq 5 \