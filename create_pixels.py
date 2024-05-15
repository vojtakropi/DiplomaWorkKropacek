import numpy as np
from PIL import Image
import cv2
import os

def main():

    root_path = ""
    source_path = root_path + "image_ribs_val"
    result_path = "image_pixel_ribs_val"

    for i in os.listdir(source_path):
        print(i)
        original = cv2.imread(os.path.join(source_path, i))
        original = cv2.resize(original, (512, 512), cv2.INTER_CUBIC)

        # Get the dimensions of the original image
        height, width, _ = original.shape

        # Create a list to store the images
        pixel_images = []

        # Iterate through each pixel of the original image
        for y in range(height):
            for x in range(width):
                # Extract the pixel value
                pixel_value = original[y, x]

                # Create a 1x1 image with the pixel value
                new_image = np.full((1, 1, 3), pixel_value, dtype=np.uint8)

                # Append the new image to the list
                pixel_images.append(new_image)

        # Convert the list of images to a numpy array
        pixel_images = np.array(pixel_images)

        # Save or do something with the pixel images
        # For example, save the images as separate files
        for t, image in enumerate(pixel_images):
            print(t)
            cv2.imwrite(result_path + "/" + i.replace(".png", "") + "_" + str(t) + ".png", image)


if __name__ == "__main__":
    main()