import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
def main():
    root_path = ""
    source_path = root_path + "D:/unetfs/UNet_ribs2"
    result_path = "ribs_noaug/val/target/"

    for i in os.listdir(source_path):
        print(i)
        original = cv2.imread(os.path.join(source_path, i), cv2.IMREAD_GRAYSCALE)

        equalized = cv2.equalizeHist(original)

        # Apply adaptive thresholding
        _, thresh = cv2.threshold(equalized, 200, 255, cv2.THRESH_BINARY)
        heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
        heatmap_on_image = cv2.addWeighted(cv2.cvtColor(original, cv2.COLOR_GRAY2BGR), 0.5, heatmap, 0.5, 0)

        # Display the heatmap overlaid on the original image
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(heatmap_on_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()