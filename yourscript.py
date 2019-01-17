import cv2
from utils.opv import OpvModel,OpvExec

mymodel = OpvModel("squeezenet1.1", device="MYRIAD", fp="FP16")
prediction = mymodel.Predict(cv2.imread("images/dog.jpeg"))

print("Verify the format of your results:")
print(prediction.shape)