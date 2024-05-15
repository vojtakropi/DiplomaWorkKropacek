import numpy as np
from PIL import Image
import os
import cv2
from cv2 import imshow, waitKey

def main():
    # Load the original image
    root_path = ""
    source_path = root_path + "image_val"
    result_path = "image_pixel_val"

    for i in os.listdir(source_path):
        print(i)
        original = cv2.imread(os.path.join(source_path, i))
        original = cv2.resize(original, (512, 512), cv2.INTER_CUBIC)

        # Get the dimensions of the original image
        height, width, depth = original.shape

        # Define the size of the neighborhood
        neighborhood_size = 31
        neighborhood_radius = neighborhood_size // 2

        # Create a list to store the images
        neighborhood_images = []

        # Iterate through each pixel of the original image
        for y in range(height):
            for x in range(width):
                # Extract the neighborhood around the current pixel
                neighborhood = original[max(0, y - neighborhood_radius):min(height, y + neighborhood_radius + 1),
                               max(0, x - neighborhood_radius):min(width, x + neighborhood_radius + 1)]

                # Create an image of size 31*31 with the neighborhood and current pixel
                new_image = np.zeros((neighborhood_size, neighborhood_size, depth))
                new_image[:neighborhood.shape[0], :neighborhood.shape[1], :neighborhood.shape[2]] = neighborhood
                # Append the new image to the list
                neighborhood_images.append(new_image)

        # Convert the list of images to a numpy array
        neighborhood_images = np.array(neighborhood_images)

        # Save or do something with the neighborhood images
        # For example, save the images as separate files
        for t, image in enumerate(neighborhood_images):
            print(t)
            cv2.imwrite(result_path + "/" + i.replace(".png", "") + "_" + str(t) + ".png", image)


if __name__ == "__main__":
    main()
