import cv2
import numpy as np
import HandTrackingModule as Htm
import time
import autopy

wCam, hCam = 640, 480
framR = 100 # frame reduce
smoothening = 6

plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

wScr, hScr = autopy.screen.size()

detector = Htm.handDetector(maxHands=1)
pTime = 0
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    cv2.rectangle(img, (framR, framR-50), (wCam - framR, hCam - framR-50), (0, 255, 0), 2)

    # get fingertips
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # check which finders are up
        fingers = detector.fingersUp()

        # only index up
        if fingers[1] == 1 and sum(fingers) == 1:


            # convert coords
            if framR<x1<wCam-framR and framR-50<y1<hCam-framR-50:
                x3 = np.interp(x1, (framR, wCam-framR), (0, wScr))
                y3 = np.interp(y1, (framR-50, hCam-framR-50), (0, hScr))

                # smoothen values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # move mouse
                autopy.mouse.move(clocX, clocY)
                plocX, plocY = clocX, clocY
            cv2.circle(img, (x1, y1), 10, (250, 250, 250), cv2.FILLED)

        # index and middle up
        if fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
            length, img, lineInfo = detector.findDistance(8,12, img)

            # click mouse if distance short
            if length<20:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (250, 250, 250), cv2.FILLED)

                autopy.mouse.click()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS:" + str(int(fps)), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(1)