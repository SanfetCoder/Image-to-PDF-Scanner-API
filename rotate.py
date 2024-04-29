from PIL import Image

def rotate_image(image_path, angle):
    """
    Rotate the image by a specified angle.

    Args:
    - image_path (str): Path to the image file.
    - angle (int): Angle of rotation in degrees (clockwise).

    Returns:
    - rotated_image (PIL.Image): Rotated image object.
    """
    # Open the image file
    image = Image.open(image_path)
    
    # Rotate the image by the specified angle (clockwise)
    rotated_image = image.rotate(angle,expand=True)
    
    return rotated_image

# Example usage
image_path = 'D:\work\scan_image\permit-scan-document\desk-result.png'  # Replace with the path to your image file
rotation_angle = -90  # Specify the angle of rotation in degrees (clockwise)
rotated_image = rotate_image(image_path, rotation_angle)
rotated_image.show()  # Display the rotated image