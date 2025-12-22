import cv2
import time
import os
import HandTrackingModule as Htm

wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = 'FingerImages'
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))

pTime = 0
detector = Htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers=[]

        # thumb
        if lmList[4][1] < lmList[17][1]:
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # other fingers
        for Id in range(1, 5):
            if lmList[tipIds[Id]][2] < lmList[tipIds[Id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        totalFinger = fingers.count(1)

        h, w, c = overlayList[totalFinger-1].shape
        img[0:h, 0:w] = overlayList[totalFinger-1]

        cv2.rectangle(img, (40,225), (140,325), (0,255,0), cv2.FILLED)
        cv2.putText(img, str(totalFinger), (60, 305), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 7)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {str(int(fps))}', (1100, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (10, 240, 20), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)