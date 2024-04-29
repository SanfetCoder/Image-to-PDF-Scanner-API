from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from io import BytesIO
import os
import numpy as np
import PIL
from PIL import Image
import cv2
import imutils
from skimage.filters import threshold_local
from helper.transform import perspective_transform,resize_to_a4
from helper.image import convert_to_cv 
import matplotlib.pyplot as plt



image_path = r"D:\work\scan_image\permit-scan-document\image\desk.JPG"
with open(image_path, "rb") as f:
    file_bytes = f.read()# Reading image from the request

image_bytes = file_bytes
image_stream = BytesIO(image_bytes) # Buffer the bytes in-memory
image = Image.open(image_stream) # Read the file bytes and store it in image variable
copy = image.copy() # Copy version of the image as original file


width, height = image.size # Get the dimension of the image
ratio = width / 500.0 # Ratio of the image
resized_image = imutils.resize(np.array(image), 500)

# Perform desired image processing here (e.g., resizing, grayscaling, etc.)
processed_image = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2GRAY) # Convert colored image to gray scale
blurred_image = cv2.GaussianBlur(processed_image, (5, 5), 0) # Applying edge detector
edged_img = cv2.Canny(blurred_image, 75, 200) # Find the edge of document

cv2.imwrite("test_4-process.png", processed_image)  # Save the processed image temporarily
cv2.imwrite("test_4-blur.png", blurred_image)  # Save the processed image temporarily
cv2.imwrite("test_4-edge.png", edged_img)  # Save the processed image temporarily


# Find the largest contour of the edges
cnts, _ = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]


for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # break this loop if approximate contours is 4
    if len(approx) == 4:
        doc = approx
        print(doc)  # Print the coordinates of the four corners
        print('finished')
        break
    else:
        print('please take a new picture to clearly see ')

# Circling the Four Corners of the Document Contour
p = []
for d in doc:
    tuple_point = tuple(d[0])
    cv2.circle(processed_image, tuple_point, 3, (0, 0, 255), 4)
    p.append(tuple_point)
# Create a blank image
img = np.zeros((300, 400, 3), dtype=np.uint8)  # Adjust dimensions as needed
# Draw the contour on the image
cv2.drawContours(img, [np.array(doc)], -1, (0, 255, 0), 2)

# # Display the image corner
# cv2.imwrite("test_4-corner.png", img)  # Save the processed image temporarily



# Warp the image 
warped_image = perspective_transform(copy, doc.reshape(4, 2) * ratio)
warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)


# cv2.imwrite("desk-warp.png", warped_image)  # Save the processed image temporarily



# Applying Adaptive Threshold and Saving the Scanned Output
T = threshold_local(warped_image, 11, offset=10, method="gaussian")
warped = (warped_image > T).astype("uint8") * 255

# Resize the thresholded image to A4 format
resized_image_a4 = resize_to_a4(warped, T)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the original image in the first subplot
axes[0].imshow(warped_image)
axes[0].set_title('warped Image')
axes[0].axis('off')

# Plot the resized image in the second subplot
axes[1].imshow(resized_image_a4)
axes[1].set_title('Resized Image')
axes[1].axis('off')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
# Convert the processed image to a BytesIO object
processed_image_bytes = BytesIO()
cv2.imwrite("temp.png", resized_image_a4)  # Save the processed image temporarily
with open("temp.png", "rb") as f:
    processed_image_bytes.seek(0)
    processed_image_bytes.truncate()
    processed_image_bytes.write(f.read())

# Get the processed image bytes from the BytesIO object
processed_image_bytes.seek(0)
final_image = processed_image_bytes.read()

# Save the processed image bytes to a file
with open("desk-result.png", "wb") as out_f:
    out_f.write(final_image)

