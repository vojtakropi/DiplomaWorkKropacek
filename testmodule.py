import os
import cv2 as cv
import tensorflow as tf


def main():
    print(tf.__version__)

    imagedir = ""
    train = "train/source"
    label =  "train/target"
    label_new = "train/target2"
    # val = "val/orig"
    # val_label = "val/label"
    # imagedir2 = "augmented2"


    for image in os.listdir(os.path.join(imagedir, train)):
        img = cv.imread(os.path.join(imagedir, train, image))
        valimg = cv.imread(os.path.join(imagedir, label, image.replace("orig", "bonesupp")))
        # img = cv.resize(img, (416, 416), cv.INTER_CUBIC)
        # valimg = cv.resize(valimg, (416, 416), cv.INTER_CUBIC)
        path = os.path.join(imagedir, train, image)
        path2 = os.path.join(imagedir, label_new, image)
        cv.imwrite(path, img)
        cv.imwrite(path2, valimg)

    # for image in os.listdir(os.path.join(imagedir, val)):
    #     img = cv.imread(os.path.join(imagedir, val, image))
    #     valimg = cv.imread(os.path.join(imagedir, val_label, image))
    #     img = cv.resize(img, (416, 416), cv.INTER_CUBIC)
    #     valimg = cv.resize(valimg, (416, 416), cv.INTER_CUBIC)
    #     path = os.path.join(imagedir2, val, image)
    #     path2 = os.path.join(imagedir2, val_label, image)
    #     cv.imwrite(path, img)
    #     cv.imwrite(path2, valimg)

if __name__ == "__main__":
    main()