# https://github.com/simpledevelopments/OpenVINO-Python-Utils
#
# OpvModel
# - name
# - input_layer
# - input_shape
# - output_layer
# - preprocess(bgr_image)   
# - predict(bgr_image)      returns list of results
# - lastresult [if prediction has been made]
# - labels     [if labels are available]
# - ClearMachine()
#
# Run ClearMachine() after the model is no longer required for inferences. Not required if executing from command line.

import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin
import os

class OpvExec:
    __machine = None
    __counter = 0
    def __init__(self):
        pass
    def _IncCount(self):
        self.__class__.__counter +=1
    def _GetCount(self):
        return self.__class__.__counter
    def _HasValidMachine(self):
        return (self.__class__.__machine != None)
    def _SetMachine(self, machine):
        self.__class__.__machine = machine
    def _GetMachine(self):
        assert(self.__class__.__machine is not None),"Please check that a valid model has been loaded"
        return self.__class__.__machine

    #For releasing loaded graph on the device (e.g. NCS2) 
    def ClearMachine(self):                 
        tmp = self.__class__.__machine
        self.__class__.__machine = None
        del tmp        
    
class OpvModel(OpvExec):
    def __init__(self, model_name, device, fp="FP32", debug=False):
        OpvExec.__init__(self)
        assert(fp in ('FP16','FP32'))
        self.name = model_name
        self._debug = debug
        if (self._GetCount()>0):
            self.ClearMachine()
            if (self._debug == True):
                print("Loaded Machine Released")
        
        #Load the .xml and .bin files for the model (and .labels if it exists)
        model_xml = "models/"+model_name+"/"+fp+"/"+model_name+".xml"
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        model_labels = os.path.splitext(model_xml)[0] + ".labels"
        if (os.path.exists(model_labels)):
            self.labels = np.loadtxt(model_labels,dtype="str",delimiter="\n")
        net = IENetwork(model=model_xml, weights=model_bin)
        net.batch_size = 1 
        
        #Initialize the hardware device (e.g. CPU / NCS2)
        plugin = IEPlugin(device=device)
        if plugin.device == "CPU":
            supported_layers = plugin.get_supported_layers(net)
            not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                print("[ERROR] These layers are not supported by the plugin for specified device {}:\n {}".
                          format(plugin.device, ', '.join(not_supported_layers)))
                del net
                del plugin
                assert(len(not_supported_layers)==0)
        
        self.input_layer = next(iter(net.inputs))
        self.input_shape = net.inputs[self.input_layer].shape
        self.output_layer = next(iter(net.outputs))
        
        self._SetMachine(plugin.load(network=net))
        self._IncCount()
        del net
        del plugin
        if (self._debug):
            print("[INFO] Model " + model_name + " Loaded and Ready")

    def Preprocess(self, image):                              # Preprocess the image
        original = image.copy()

        (n, c, h, w) = self.input_shape
        images = np.ndarray(shape=(n, c, h, w))
        if image.shape[:-1] != (h, w):
            if (self._debug):
                print("\t[INFO] Image resized from {} to {}".format(image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        images[0] = image.transpose((2, 0, 1))               # Change data layout from HWC to CHW
        return (original, images)

    def Predict(self, image, layer=None):
        if (layer==None):
            layer = self.output_layer
        (self.original,image) = self.Preprocess(image)
        
        self.lastresult = self._GetMachine().infer(inputs={self.input_layer: image}) #for models with more than 1 output layer
        output = self.lastresult[layer]
        return output                                        # Return the default output layer
        
