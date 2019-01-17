# OpenVINO-Python-Utils
Helper Class for OpenVINO Python - tested on 2018R5


This is an attempt to make it easier to run inferences using OpenVINO in Python

## Directory Structure:
Download the folder, rename it to utils, then place it in the same folder as your python script. Place models inside the models subdirectory. See example below:
```
yourscript.py
|- utils
|  |- __init__.py
|  |- opv.py
|- images
|  |- dog.jpeg
|  |- boy.jpeg
|  |- truck.jpeg
|- models
   |- squeezenet1.1
      |- FP16
      |  |- squeezenet1.1.xml
      |  |- squeezenet1.1.bin
      |  |- squeezenet1.1.labels
      |- FP32
         |- squeezenet1.1.xml
         |- squeezenet1.1.bin
         |- squeezenet1.1.labels
```
## Sample Code
Assuming you have already set up the python bindings for openvino.inference_engine (ships with OpenVINO), 
you can run the sample code below. Instead of squeezenet1.1, you can other pre-trained models that ship with the OpenVINO installation.
Simply copy the respective models from "Intel_Models" into the models subdirectory following the same format as above.
```
import cv2
from utils.opv import OpvModel,OpvExec

mymodel = OpvModel("squeezenet1.1", device="MYRIAD", fp="FP16")
prediction = mymodel.Predict(cv2.imread("images/dog.jpeg"))
```

The results stored in the variable "prediction" (in the example above) is a numpy array. 

If using the Intel pre-trained models, <br />
prediction.shape should match the documentation for the output of the respective models.

Create your own methods/functions to parse the output.


Use with caution. Do note that this code may no longer work when Intel updates the Python API. 
https://software.intel.com/en-us/articles/OpenVINO-InferEngine#overview-of-inference-engine-python-api

According to the site as at 1 January 2019, "This is a preview version of the Inference Engine Python* API for evaluation purpose only. Module structure and API itself will be changed in future releases."
