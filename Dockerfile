FROM continuumio/anaconda3
COPY . .
RUN conda create --name my_env python=3.7
ENV PATH /opt/conda/envs/my_env/bin:$PATH
RUN /bin/bash -c "source activate my_env"
RUN pwd 
RUN python -m pip install -r requirements.txt
RUN python -m pip install -e .
RUN python ./Examples/trade/train_agent.py


