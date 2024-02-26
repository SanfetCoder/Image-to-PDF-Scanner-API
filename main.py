from fastapi import FastAPI, UploadFile, File, Response
from io import BytesIO
import os
import numpy as np
from PIL import Image
import cv2
import cv2
import imutils
from skimage.filters import threshold_local

app = FastAPI()

@app.post("/image/pdf")
async def process_image(file: UploadFile = File(...)):
  image_bytes = await file.read()
  image_stream = BytesIO(image_bytes)
  image = Image.open(image_stream)
  # Perform desired image processing here (e.g., resizing, grayscaling, etc.)
  processed_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY) # Convert colored image to gray scale
  processed_image = cv2.resize(processed_image, (220, 220)) # Resize the image
  # Applying edge detector
  blurred_image = cv2.GaussianBlur(processed_image, (5, 5), 0)
  edged_img = cv2.Canny(blurred_image, 75, 200)
  cv2.imshow('Image edges', edged_img)
  cv2.waitKey(0)
  # Convert the processed image to a BytesIO object
  processed_image_bytes = BytesIO()
  cv2.imwrite("temp.png", processed_image)  # Save the processed image temporarily
  with open("temp.png", "rb") as f:
    processed_image_bytes.seek(0)
    processed_image_bytes.truncate()
    processed_image_bytes.write(f.read())
  os.remove("temp.png")  # Remove temporary file
  # Final image to be send back to the request
  final_image = processed_image_bytes.getvalue()
  return Response(content=final_image, media_type="image/png")