FROM continuumio/anaconda3


COPY . .
# Result folders creation
RUN mkdir -p ./Examples/trade/Results/Models/
RUN mkdir -p ./Examples/simpletod/Results/Models/
RUN mkdir -p ./Examples/trade/Results/baseline/
RUN mkdir -p ./Examples/simpletod/Results/baseline/
RUN mkdir -p ./Examples/trade/trade_dst/data/MultiWOZ_2.1/
RUN mkdir -p ./Examples/trade/Results/Evaluation/
RUN mkdir -p ./Examples/simpletod/Results/Evaluation/

# Copy needed models : pre-trained Trade, pre-trained simpletod

COPY Examples/trade/trade_dst/baseline/TRADE-multiwozdst/HDD400BSZ32DR0.2ACC-0.5342  ./Examples/trade/trade/baseline/TRADE-multiwozdst/HDD400BSZ32DR0.2ACC-0.5342
COPY Examples/trade/trade_dst/data/MultiWOZ_2.1/trade_ontology.json ./Examples/trade/trade_dst/data/MultiWOZ_2.1/trade_ontology.jso
COPY Examples/simpletod/simpletod/baseline/checkpoint-825000  ./Examples/simpletod/simpletod/baseline/checkpoint-825000

RUN conda create --name my_env python=3.7
ENV PATH /opt/conda/envs/my_env/bin:$PATH
RUN /bin/bash -c "source activate my_env"

RUN python -m pip install -r requirements.txt
RUN python -m pip install -e .

# just an example
CMD python ./Examples/trade/train_agent.py
ENTRYPOINT ["python", "trainer/mnist.py"]

