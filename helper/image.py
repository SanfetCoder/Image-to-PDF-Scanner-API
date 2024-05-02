import cv2
import PIL
import numpy as np
import os


def convert_to_cv(image : PIL.Image):
  return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def is_heic_file(file_path):
    # Get the file extension
    _, file_extension = os.path.splitext(file_path)
    # Check if the file extension indicates a HEIC file
    return file_extension.lower() in ['.heic', '.heif']