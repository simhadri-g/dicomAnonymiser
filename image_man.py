import pydicom as dicom
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import dicom as pdicom
import numpy as np
import time
from imutils.object_detection import non_max_suppression
import pytesseract
import imutils
from PIL import Image

def get_image(path,width):

    ds = dicom.dcmread(path)
    d = ds.pixel_array

    plt.imsave("./temp/read.png", d,cmap = plt.cm.bone)


    #cv2.imwrite("ll.png",d)
    img = cv2.imread("./temp/read.png",1)
    img = imutils.resize(img, width=width)
    cv2.imwrite("./temp/read_display.png",img)


    return img

def detect(img):

    image = img
    args = dict()

    args["east"] = "./Resources/frozen_east_text_detection.pb"
    args["min_confidence"] = 0.1
    args["width"] = image.shape[0]
    args["height"] = image.shape[1]

    orig = image.copy()
    (H, W) = image.shape[:2]

    (newW, newH) = (args["width"], args["height"])
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]

    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(args["east"])

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < args["min_confidence"]:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    colour = (0, 0, 0)
    orig1 = orig.copy()
    # loop over the bounding boxes
    bb=dict()
    print("boxes =", len(boxes))
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # draw the bounding box on the image
        ph = 5  # 10
        pv = 2  # 5

        t=tuple()
        if (endX < orig.shape[0] / 2):
            t = (startY, endY, 0, endX)
            imgcrop = orig[startY:endY, 0:endX]

        elif (startX > orig.shape[0] / 2):
            t = (startY,endY, startX,orig.shape[1])
            imgcrop = orig[startY:endY, startX:orig.shape[1]]

        else:
            t = (startY,endY, startX,endX)
            imgcrop = orig[startY:endY, startX:endX]

        text = pytesseract.image_to_string(imgcrop, lang="eng", config='--psm 8')

        bb[text] = (t,imgcrop)

    print(sorted(bb,reverse= True))
    cv2.imwrite("./temp/fin.png", orig1)

    return sorted(bb,reverse= True),bb


def mask(bbox):

    img = cv2.imread("./temp/read.png",1)
    for box in bbox:
        img[box[0]:box[1],box[2]:box[3]] = (0,0,0)

    cv2.imwrite("./temp/mask.png",img)

    #cv2.imshow("mask",img)

def full_mask():
    image = cv2.imread("./temp/read.png",1)

    args = dict()

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", type=str,help="path to input image")
    args[
        "east"] = "/home/roopak1997/python_projects/Gehack/frozen_east_text_detection.pb"  # ap.add_argument("-east", "--east", type=str,help="path to input EAST text detector")
    args[
        "min_confidence"] = 0.1  # ap.add_argument("-c", "--min-confidence", type=float, default=0.5,help="minimum probability required to inspect a region")
    args["width"] = image.shape[
        0]  # ap.add_argument("-w", "--width", type=int, default=320,help="resized image width (should be multiple of 32)")
    args["height"] = image.shape[
        1]  # ap.add_argument("-e", "--height", type=int, default=320,help="resized image height (should be multiple of 32)")
    # args = vars(ap.parse_args())

    # load the input image and grab the image dimensions
    # image = cv2.imread(args["image"])
    orig = image.copy()
    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (args["width"], args["height"])
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(args["east"])

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < args["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    pil_im = Image.fromarray(orig)
    cc = pil_im.getcolors()

    # cc = sorted(cc,key=lambda x:x[0])
    # colour = cc[-1][1]
    colour = (0, 0, 0)
    orig1 = orig.copy()
    # loop over the bounding boxes
    bb = dict()
    print("boxes =", len(boxes))
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        if (endX < orig.shape[0] / 2):
            orig[startY:endY, 0:endX] = colour



        elif (startX > orig.shape[0] / 2):
            orig[startY:endY, startX:orig.shape[1]] = colour


        else:
            orig[startY:endY, startX:endX] = colour

    img=orig
    imgcrop = orig
    gray = cv2.cvtColor(imgcrop, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    # cv2.imshow("bw",gray)
    blur = cv2.blur(gray, (3, 3))  # blur the image
    # cv2.imshow("blur", blur)
    ret, thresh = cv2.threshold(blur, 12, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hull = []

    contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    p = 10

    drawing = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    for cont in contours_sorted:
        # print(cv2.contourArea(cont),0.0003*img.shape[1]*img.shape[0])
        if (cv2.contourArea(cont) > (0.005 * img.shape[1] * img.shape[0])):
            x, y, w, h = cv2.boundingRect(cont)
            x -= p
            y -= p
            w += p
            h += p
            # cv2.rectangle(imgcrop, (x, y), (x + w, y + h), (0, 255, 255), 2)
            drawing[y:y + h, x:x + w] = img[y:y + h, x:x + w]


    cv2.imwrite("./temp/full_mask.png",drawing)
    return drawing


