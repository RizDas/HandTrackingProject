import cv2
import time
import numpy as np
import HandTrackingModule as Htm
import math
from pycaw.pycaw import AudioUtilities

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = Htm.handDetector(detectionCon=0.7)

device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
#print(f"Audio output: {device.FriendlyName}")
#print(f"- Muted: {bool(volume.GetMute())}")
#print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
volRange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-6.0, None)

minVol = volRange[0]
maxVol = volRange[1]
vol=0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1, y1), 10, (200, 10, 50), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (200, 10, 50), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (200, 10, 50), 2)
        cv2.circle(img, (cx, cy), 7, (200, 10, 50), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        # print(length)

        # Hand Range: 40 - 300
        # Volume Range: -65 - 0

        vol = np.interp(length, [40, 200], [minVol, maxVol])
        volBar = np.interp(length, [40, 200], [400, 120])
        volPer = np.interp(length, [40, 200], [0, 100])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 40:
            cv2.circle(img, (cx, cy), 7, (180, 150, 150), cv2.FILLED)

        if length > 200:
            cv2.circle(img, (cx, cy), 7, (40, 200, 50), cv2.FILLED)

    cv2.rectangle(img, (50, 120), (65, 400), (14, 13, 15), cv2.FILLED)
    cv2.rectangle(img, (50, int(volBar)), (65, 400), (240, 240, 240), cv2.FILLED)
    cv2.putText(img, f'{str(int(volPer))} %', (45, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 240, 20), 2)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {str(int(fps))}', (40, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (10, 240, 20), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)