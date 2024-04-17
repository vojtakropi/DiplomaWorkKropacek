import os
import cv2
import numpy as np

def main():
    root_path = ""
    source_path = root_path + "train/"
    source = 'source'
    target = 'target'
    result_path = 'subtracted'

    for i in os.listdir(os.path.join(source_path, source)):
        original = cv2.imread(os.path.join(source_path, source, i))
        final = cv2.imread(os.path.join(source_path, target, i.replace("orig", "bonesupp")))
        subtracted = cv2.subtract(original, final)
        alpha = 3.5
        new_image = np.clip(alpha * subtracted, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(result_path, i), new_image)




if __name__ == "__main__":
    main()