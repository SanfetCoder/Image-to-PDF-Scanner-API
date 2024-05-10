import numpy as np
import cv2
from PIL import Image
from helper.image import convert_to_cv
from io import BytesIO 
from cv2 import dnn_superres



def order_points(pts):
	# initialzie a list of coordinates that will be ordered
  # such that the first entry in the list is the top-left,
  # the second entry is the top-right, the third is the
  # bottom-right, and the fourth is the bottom-left
  rect = np.zeros((4, 2), dtype = "float32")

  # the top-left point will have the smallest sum, whereas
  # the bottom-right point will have the largest sum
  s = pts.sum(axis = 1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]

  # now, compute the difference between the points, the
  # top-right point will have the smallest difference,
  # whereas the bottom-left will have the largest difference
  diff = np.diff(pts, axis = 1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]

  # return the ordered coordinates
  return rect


def perspective_transform(image, pts):
	# obtain a consistent order of the points and unpack them
  # individually
  rect = order_points(pts)
  (tl, tr, br, bl) = rect

  # compute the width of the new image, which will be the
  # maximum distance between bottom-right and bottom-left
  # x-coordiates or the top-right and top-left x-coordinates
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))

  # compute the height of the new image, which will be the
  # maximum distance between the top-right and bottom-right
  # y-coordinates or the top-left and bottom-left y-coordinates
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))

  # now that we have the dimensions of the new image, construct
  # the set of destination points to obtain a "birds eye view",
  # (i.e. top-down view) of the image, again specifying points
  # in the top-left, top-right, bottom-right, and bottom-left
  # order
  dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")

  # compute the perspective transform matrix and then apply it
  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(convert_to_cv(image), M, (maxWidth, maxHeight))

  # return the warped image
  return warped

def resize_to_a4(warped_image, T, dpi=400):
    # Define A4 paper dimensions in inches
    a4_width_inch = 8.27
    a4_height_inch = 11.69

    # Convert A4 dimensions from inches to pixels
    a4_width_pixels = int(a4_width_inch * dpi)
    a4_height_pixels = int(a4_height_inch * dpi)

    # Get the dimensions of the original image
    orig_height, orig_width = warped_image.shape[:2]

    # Calculate the aspect ratio of the original image
    orig_aspect_ratio = orig_width / orig_height

    # Calculate the target width and height for resizing while maintaining aspect ratio
    if orig_aspect_ratio > 1:  # If the original image is horizontal
        target_width = a4_width_pixels
        target_height = int(target_width / orig_aspect_ratio)
    else:  # If the original image is vertical
        target_height = a4_height_pixels
        target_width = int(target_height * orig_aspect_ratio)

    # Resize the warped image to the target dimensions
    resized_warped = (warped_image > T).astype("uint8") * 255
    resized_warped = cv2.resize(resized_warped, (target_width, target_height))

    return resized_warped


def convert_to_png(image):
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        png_data = BytesIO()  # Create a BytesIO object to hold the PNG data
        image.save(png_data, format='PNG')  # Save the image as PNG into the BytesIO object
        png_data.seek(0)  # Move the cursor to the beginning of the BytesIO object
        print("Image converted successfully!")
        return png_data
    except Exception as e:
        print(f"An error occurred: {e}")

# some image might need rotate this code might help you
def rotate_image(image_path, angle):
    # Open the image file
    image = Image.open(image_path)
    
    # Rotate the image by the specified angle (clockwise)
    rotated_image = image.rotate(angle,expand=True)
    
    return rotated_image
  # # Example usage
  # image_path = r'D:\work\scan_image\permit-scan-document\temp.png'  # Replace with the path to your image file
  # rotation_angle = -90  # Specify the angle of rotation in degrees (clockwise)
  # rotated_image = rotate_image(image_path, rotation_angle)
  # rotated_image.show()  # Display the rotated image


def upscale(input_image):
  # Create an SR object
  sr = dnn_superres.DnnSuperResImpl_create()

  # Read image
  image = input_image
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
  result = Image.fromarray(result)
 
  return result