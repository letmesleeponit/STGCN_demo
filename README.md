# STGCN inference

## Guide
This repository demonstrates how to use Docker to perform inference using the STGCN model.  
Please follow the steps below to try it out.  
By completing the final step, you will obtain the output.mp4 file, which represents the inference result of vid.mp4.
![image](output.gif)

## Step
1. Build the docker image.
```
docker image build --tag stgcn_demo .
```
2. Create a Docker container named STGCN_demo with GPU support and bind the current directory to the container:
```
docker container run -it --gpus all --name STGCN_demo --mount type=bind,source="$(pwd)",target="/STGCN_demo" stgcn_demo /bin/bash
```
3. Activate specific enviroment.
```
conda activate lightning
```
4. Clone the STGCN repository.
```
git clone https://github.com/yysijie/st-gcn
```
5. Adjust the permission settings to allow access.
```
chmod -R 777 st-gcn
```
6. Modify the st-gcn/processor/io.py file as follows:
```
# Original:
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

# Revised:
from torchlight import torchlight
from torchlight.torchlight.io import str2bool
from torchlight.torchlight.io import DictAction
from torchlight.torchlight.io import import_class
```
7. Replace the graph setting of the human body.
```
paste graph.py to st-gcn/net/utils/graph.py
```
8. Reinstall opencv-python-headless package.
```
pip uninstall opencv-python-headless
pip install opencv-python-headless==4.6.0.66
```
9. Running the demo.py script in the terminal.
```
python demo.py
```

## Citation
[STGCN](https://github.com/yysijie/st-gcn)