import os
import cv2
import numpy as np

def main():
    root_path = ""
    source_path = root_path + "D:/unetfs/UNet_ribs2"
    result_path = "ribs_noaug/val/source"
    newpath = "substracted_unetribs"
    nep = "test50"

    for i in os.listdir(newpath):
        print(i)
        original = cv2.imread(os.path.join(newpath, i))
        target = cv2.imread(os.path.join(result_path, i.replace("_pred", "")))
        subtracted = cv2.subtract(target, original)
        cv2.imwrite(os.path.join(nep, i), subtracted)


if __name__ == "__main__":
    main()