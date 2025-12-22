import cv2
import numpy as np
import time
import os
import HandTrackingModule as Htm

brushThickness = 15
eraserThickness = 80

folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
drawColour = (255, 182, 56)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = Htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # check fingers up

        fingers = detector.fingersUp()
        # print(fingers)

        # selection mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("drawing")

            if y1 < 150:
                if 285<x1<435:
                    header = overlayList[0]
                    drawColour = (255, 182, 56)
                elif 470<x1<620:
                    header = overlayList[1]
                    drawColour = (196, 102, 255)
                elif 655<x1<805:
                    header = overlayList[2]
                    drawColour = (87, 217, 126)
                elif 840<x1<990:
                    header = overlayList[3]
                    drawColour = (89, 222, 255)
                elif 1050<x1<1250:
                    header = overlayList[4]
                    drawColour = (0, 0, 0)

            cv2.line(img, (x1, y1), (x2, y2), drawColour, 5)


        # drawing mode
        if fingers[1] and not fingers[2]:
            if drawColour == (0, 0, 0):
                cv2.circle(img, (x1, y1), 40, drawColour, cv2.FILLED)
            else:
                cv2.circle(img, (x1, y1), 15, drawColour, cv2.FILLED)
            print("drawing")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColour == (0,0,0):
                # cv2.line(img, (xp, yp), (x1, y1), drawColour, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColour, eraserThickness)
            else:
                # cv2.line(img, (xp, yp), (x1, y1), drawColour, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColour, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInverse = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInverse)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:150, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    # cv2.rectangle(img, (285, 0), (435, 150), (0, 255, 0), cv2.FILLED)
    # cv2.rectangle(img, (470, 0), (620, 150), (0, 255, 0), cv2.FILLED)
    # cv2.rectangle(img, (655, 0), (805, 150), (0, 255, 0), cv2.FILLED)
    # cv2.rectangle(img, (840, 0), (990, 150), (0, 255, 0), cv2.FILLED)
    # cv2.rectangle(img, (1050, 0), (1250, 150), (0, 255, 0), cv2.FILLED)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
