from io import BytesIO
import numpy as np
from PIL import Image
import cv2
import imutils
from skimage.filters import threshold_local
from transform import perspective_transform,resize_to_a4, convert_to_png
from image import is_heic_file
import matplotlib.pyplot as plt
from pillow_heif import register_heif_opener

def get_scanned_document(image_path : str):
    try:
        # Enable reading .HEIF file
        register_heif_opener()
        
        # Reading the image based on whether it is .HEIC file or not
        if is_heic_file(image_path):
            image = Image.open(image_path)
            image = convert_to_png(image)
            image = Image.open(image)
        else:
            with open(image_path, "rb") as f:
                file_bytes = f.read()
            image_bytes = file_bytes
            image_stream = BytesIO(image_bytes) 
            image = Image.open(image_stream)
            
        copy = image.copy()

        # Show the RGB image
        # plt.imshow(image)
        # plt.axis('off')
        # plt.show()

        # Normal will be <class 'PIL.Image.Image'>
        width, height = image.size # Get the dimension of the image
        ratio = width / 500.0 # Ratio of the image
        resized_image = imutils.resize(np.array(image), 500)

        # Perform desired image processing here (e.g., resizing, grayscaling, etc.)
        processed_image = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2GRAY) # Convert colored image to gray scale
        blurred_image = cv2.GaussianBlur(processed_image, (5, 5), 0) # Applying edge detector
        edged_img = cv2.Canny(blurred_image, 75, 200) # Find the edge of document

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
            else:
                raise Exception("Can't find the contours of this image. Please try again!")

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

        # Warp the image 
        warped_image = perspective_transform(copy, doc.reshape(4, 2) * ratio)
        warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

        # cv2.imwrite("desk-warp.png", warped_image)  # Save the processed image temporarily

        # Applying Adaptive Threshold and Saving the Scanned Output
        T = threshold_local(warped_image, 11, offset=10, method="gaussian")
        warped = (warped_image > T).astype("uint8") * 255

        # Resize the thresholded image to A4 format
        resized_image_a4 = resize_to_a4(warped, T)

        # Convert the processed image to a BytesIO object
        processed_image_bytes = BytesIO()
        cv2.imwrite("../static/image/result/temp.png", resized_image_a4)  # Save the processed image temporarily
        with open("../static/image/result/temp.png", "rb") as f:
            processed_image_bytes.seek(0)
            processed_image_bytes.truncate()
            processed_image_bytes.write(f.read())

        # Get the processed image bytes from the BytesIO object
        processed_image_bytes.seek(0)
        final_image = processed_image_bytes.read()
        
        return final_image
    except Exception as e:
        raise Exception(e)
