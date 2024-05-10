import numpy as np
from io import BytesIO
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from cv2 import dnn_superres



# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
image = cv2.imread(r'D:\work\scan_image\permit-scan-document\image\document-scanner-original-image.png')
print(type(image))
image = Image.fromarray(image)
print(type(image))
image = np.array(image)
print(type(image))
# Read the desired model
# fast
path = "FSRCNN_x4.pb" 
# # medium
# path = "LapSRN_x8.pb" 
# # slow
# path = "EDSR_x4.pb" 
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("fsrcnn", 4)

# Upscale the image
result = sr.upsample(image)



