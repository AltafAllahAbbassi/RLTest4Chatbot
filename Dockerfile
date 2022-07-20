FROM continuumio/anaconda3
COPY . .
RUN conda create --name my_env python=3.7
ENV PATH /opt/conda/envs/my_env/bin:$PATH
RUN /bin/bash -c "source activate my_env"
RUN pwd 
RUN python -m pip install -r requirements.txt
RUN python -m pip install -e .

# Result folder creation
RUN mkdir -p ./Examples/trade/Results/Models/
RUN mkdir -p ./Examples/simpletod/Results/Models/

RUN mkdir -p ./Examples/trade/Results/baseline/
RUN mkdir -p ./Examples/simpletod/Results/baseline/

RUN mkdir -p ./Examples/trade/Results/Evaluation/
RUN mkdir -p ./Examples/simpletod/Results/Evaluation/

# Copy needed models : pre-trained Trade, pre-trained simpletod

COPY Examples/trade/trade_dst/baseline/TRADE-multiwozdst/HDD400BSZ32DR0.2ACC-0.5342  ./Examples/trade/trade/baseline/TRADE-multiwozdst/HDD400BSZ32DR0.2ACC-0.5342
COPY Examples/trade/trade/data/MultiWOZ_2.1/trade_ontology.json ./Examples/trade/tradedata/MultiWOZ_2.1/trade_ontology.json

COPY Examples/simpletod/simpletod/baseline/checkpoint-825000  ./Examples/simpletod/simpletod/baseline/checkpoint-825000

# just an exampe 
RUN python ./Examples/trade/train_agent.py


