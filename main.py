from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from io import BytesIO
import os
import numpy as np
from PIL import Image
import cv2
import imutils
from skimage.filters import threshold_local
from helper.transform import perspective_transform
from helper.image import convert_to_cv

# App instance
app = FastAPI()

@app.post("/image/pdf")
async def process_image(file: UploadFile = File(...)):
  try:
    # Reading image from the request
    image_bytes = await file.read() # Read some bytes from the file
    image_stream = BytesIO(image_bytes) # Buffer the bytes in-memory
    image = Image.open(image_stream) # Read the file bytes and store it in image variable
    cv2.imshow('Original Image', convert_to_cv(image)) # Show the image to the screen
    cv2.waitKey(0) # Wait for users to click any key to close down the showing window
    copy = image.copy() # Copy version of the image as original file
    # Image dimension
    width, height = image.size # Get the dimension of the image
    ratio = width / 500.0 # Ratio of the image
    resized_image = imutils.resize(np.array(image), 500)
    cv2.imshow("Resized image", resized_image)
    cv2.waitKey(0)
    # Perform desired image processing here (e.g., resizing, grayscaling, etc.)
    processed_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY) # Convert colored image to gray scale
    blurred_image = cv2.GaussianBlur(processed_image, (5, 5), 0) # Applying edge detector
    edged_img = cv2.Canny(blurred_image, 75, 200) # Find the edge of document
    cv2.imshow("Edged image", edged_img)
    cv2.waitKey(0)
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
    cv2.imshow("Warped", imutils.resize(warped_image, height = 650))
    cv2.waitKey(0)
    # Applying Adaptive Threshold and Saving the Scanned Output
    T = threshold_local(warped_image, 11, offset=10, method="gaussian")
    warped = (warped_image > T).astype("uint8") * 255
    cv2.imwrite('./'+'scan'+'.png',warped)
    # Output image
    # processed_image = imutils.resize(warped, 650)
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
  except Exception as error:
    print(error)
    raise HTTPException(status_code=500, detail="There was an error while trying to process the image")