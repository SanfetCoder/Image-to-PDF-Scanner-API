from helper.transform import perspective_transform,resize_to_a4,convert_to_png,rotate_image
from io import BytesIO
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()



image_path = r"D:\work\scan_image\permit-scan-document\image\document-scanner-original-image.heic"
image = Image.open(image_path)

image = convert_to_png(image)
print(type(image))

image = Image.open(image)
print(type(image))

# image.show()





