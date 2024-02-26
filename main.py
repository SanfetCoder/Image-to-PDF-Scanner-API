from fastapi import FastAPI, UploadFile, File, Response
from io import BytesIO
import os
import numpy as np
from PIL import Image
import cv2
import cv2
import imutils
from skimage.filters import threshold_local
from helper.transform import perspective_transform, get_image_dimensions

app = FastAPI()

@app.post("/image/pdf")
async def process_image(file: UploadFile = File(...)):
  image_bytes = await file.read()
  image_stream = BytesIO(image_bytes)
  image = Image.open(image_stream)
  # Copy version of the image
  copy = image.copy()
  # Get the dimension of the image
  image_width, image_height = get_image_dimensions(image)
  # Ratio of the image
  ratio = image_width / 500.0
  # Perform desired image processing here (e.g., resizing, grayscaling, etc.)
  processed_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY) # Convert colored image to gray scale
  processed_image = cv2.resize(processed_image, (220, 220)) # Resize the image
  # Applying edge detector
  blurred_image = cv2.GaussianBlur(processed_image, (5, 5), 0)
  edged_img = cv2.Canny(blurred_image, 75, 200)
  # Find the largest contour of the edges
  cnts, _ = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
  for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # break this loop if approximate contours is 4
    if len(approx) == 4:
      doc = approx
      break
  # Circling the Four Corners of the Document Contour
  p = []
  for d in doc:
    tuple_point = tuple(d[0])
    cv2.circle(processed_image, tuple_point, 3, (0, 0, 255), 4)
    p.append(tuple_point)
  # Warp the image 
  warped_image = perspective_transform(copy, doc.reshape(4, 2) * ratio)
  warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
  cv2.imshow("Warped Image", imutils.resize(warped_image, height=650))
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