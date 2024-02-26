import cv2
import PIL
import numpy as np

def convert_to_cv(image : PIL.Image):
  return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)