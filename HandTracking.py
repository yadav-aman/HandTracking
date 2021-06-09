import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, maxhands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode
        self.maxHands = maxhands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            self.mode, self.maxHands, self.min_detection_confidence, self.min_tracking_confidence)
        self.tipIds = [4, 8, 12, 16, 20]

    def detectHands(self, img, draw=True):
        # mediapipe uses RGB images only
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False
        self.results = self.hands.process(imgRGB)
        # print(self.results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLandMarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        img, handLandMarks, self.mp_hands.HAND_CONNECTIONS)

    def findPosition(self, img, handNum=0, draw=True):
        self.posList = []
        xList = []
        yList = []
        bBox = []
        if self.results.multi_hand_landmarks:
            handLandMarks = self.results.multi_hand_landmarks[handNum]
            for lm, pos in enumerate(handLandMarks.landmark):
                # print(lm, pos)
                # getting height, width and channel of image
                h, w, _ = img.shape
                cx, cy = int(pos.x * w), int(pos.y * h)
                # print(lm, cx, cy)
                xList.append(cx)
                yList.append(cy)
                self.posList.append([lm, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15,
                               (255, 0, 255), cv2.FILLED)
        if xList and yList:
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.posList, bBox

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.posList[p1][1:]
        x2, y2 = self.posList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, [x1, y1, x2, y2, cx, cy]

    def fingersUp(self):
        fingers = []
        # Thumb - only for right hand (on inverted image)
        if self.posList[self.tipIds[0]][1] < self.posList[self.tipIds[0] - 1][1]:
            fingers.append(True)
        else:
            fingers.append(False)

        # Fingers
        for id in range(1, 5):
            if self.posList[self.tipIds[id]][2] < self.posList[self.tipIds[id] - 2][2]:
                fingers.append(True)
            else:
                fingers.append(False)
            # totalFingers = fingers.count(True)

        return fingers


def main():
    previousTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector(maxhands=1, min_detection_confidence=0.7)

    while cap.isOpened():
        success, img = cap.read()
        # Flip the image horizontally for a later selfie-view display
        img = cv2.flip(img, 1)

        detector.detectHands(img)
        posList = detector.findPosition(img, draw=False)
        # if len(posList) != 0:
        # print(posList)

        # showing FPS
        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 40),
                    cv2.QT_FONT_NORMAL, 1, (255, 0, 255), 3)

        cv2.imshow("Hand Tracking", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
