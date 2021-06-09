import cv2
import HandTracking as ht
import time
import numpy as np
import autopy

wCam, hCam = 640, 480
wScreen, hScreen = autopy.screen.size()
previousTime = 0
frameReduction = 100

smoothening = 7
pX, pY = 0, 0
cX, cY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = ht.handDetector(maxhands=1, min_detection_confidence=0.75)

while cap.isOpened():
    success, img = cap.read()

    # Flip the image horizontally for a later selfie-view display
    img = cv2.flip(img, 1)

    detector.detectHands(img)
    posList, bBox = detector.findPosition(img, draw=False)

    cv2.rectangle(img, (frameReduction, 0),
                  (wCam-frameReduction, hCam-frameReduction*2), (255, 0, 255), 2)
    if posList:
        # get the position of index and middle finger
        x1, y1 = posList[8][1:]
        x2, y2 = posList[12][1:]

        fingersUp = detector.fingersUp()
        x3 = np.interp(x1, (frameReduction, wCam -
                            frameReduction), (0, wScreen))
        y3 = np.interp(y1, (0, hCam -
                            frameReduction*2), (0, hScreen))
        # smoothening
        cX = pX + (x3 - pX) / smoothening
        cY = pY + (y3 - pY) / smoothening
        pX, pY = cX, cY
        try:
            autopy.mouse.move(cX, cY)
        except:
            print(cX, cY)

    # showing FPS
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 40),
                cv2.QT_FONT_NORMAL, 1, (255, 0, 255), 3)

    cv2.imshow("Gesture", img)
    cv2.waitKey(1)
