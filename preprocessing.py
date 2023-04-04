import random
from PIL import Image

def crop_and_split(image_path):
    # Open the image
    img = Image.open(image_path)

    # Get the image size
    width, height = img.size

    # Define the box to crop
    crop_box = (0, int(0.4 * height), width, height)

    # Crop the image
    cropped_img = img.crop(crop_box)

    # Get the cropped image size
    cropped_width, cropped_height = cropped_img.size

    # Define the boxes to split the cropped image in half
    box1 = (0, 0, cropped_width, cropped_height//2)
    box2 = (0, cropped_height//2, cropped_width, cropped_height)

    # Split the cropped image in half
    left_img = cropped_img.crop(box1)
    right_img = cropped_img.crop(box2)

    return left_img, right_img
    # Save the two halves as files
    half1.save('half1.jpg')
    half2.save('half2.jpg')


def guided_crop(image_array):
    """
    Randomly crops an input image into rectangles that are 90% of the width and height of the image.
    """

    # Get the width and height of the image
    width, height = image_array.size

    # Calculate the crop dimensions
    crop_width = int(0.9 * width)
    crop_height = int(0.9 * height)

    # Calculate the maximum x and y coordinates for the top-left corner of the crop
    max_x = width - crop_width
    max_y = height - crop_height

    # Choose a random x and y coordinate for the top-left corner of the crop
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # Crop the image using the chosen coordinates and dimensions
    crop = image_array.crop((x, y, x + crop_width, y + crop_height))

    # Return the cropped image
    return crop


# Example usage
left_image, right_image = crop_and_split('input_image.jpg')

for i in range(6):
    cropped_left_image = guided_crop(left_image)
    cropped_right_image = guided_crop(right_image)
    
    cropped_left_image.save(f"cropped_left_image_{i}.png")
    cropped_right_image.save(f"cropped_right_image_{i}.png")
