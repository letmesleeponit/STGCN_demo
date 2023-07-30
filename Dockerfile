From pytorchlightning/pytorch_lightning:base-conda-py3.8-torch1.8

WORKDIR /STGCN_demo
RUN apt-get update 
RUN apt-get install -y git zip sudo libx11-6 build-essential ca-certificates wget curl tmux htop nano vim
# libsm6 libxext6 libxrender-dev
# RUN apt install -y libgl1-mesa-glx
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN pip uninstall -y opencv-contrib-python

