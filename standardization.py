import os
import cv2
import numpy as np



def main():
    root_path = ""
    source_path = root_path + "predictions/512/UNet_ribs/"
    result_path = 'predictions_standard/'

    for i in os.listdir(source_path):
        original = cv2.imread(os.path.join(source_path, i))
        print(i)
        p = np.zeros(shape=original.shape)
        p = cv2.normalize(original, p, alpha=255, beta=-5, norm_type=cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(result_path, i), p)


if __name__ == "__main__":
    main()