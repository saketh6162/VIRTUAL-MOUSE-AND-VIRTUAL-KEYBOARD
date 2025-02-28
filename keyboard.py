Jimport cv2
import numpy as np
import time
from pynput.keyboard import Controller, Key as PynputKey
import mediapipe as mp

class HandTracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = bool(mode)
        self.maxHands = int(maxHands)
        self.detectionCon = float(detectionCon)
        self.trackCon = float(trackCon)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLm in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLm, self.mpHands.HAND_CONNECTIONS)
        return img

    def getPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

    def isLeftHand(self):
        if self.results.multi_handedness:
            return self.results.multi_handedness[0].classification[0].label == 'Left'
        return False

class Key:
    def __init__(self, x, y, w, h, text):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text

    def drawKey(self, img, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.5, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
        bg_rec = img[self.y:self.y + self.h, self.x:self.x + self.w]
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1 - alpha, 1.0)
        img[self.y:self.y + self.h, self.x:self.x + self.w] = res
        text_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)[0]
        text_pos = (self.x + self.w // 2 - text_size[0] // 2, self.y + self.h // 2 + text_size[1] // 2)
        cv2.putText(img, self.text, text_pos, fontFace, fontScale, text_color, thickness)

    def isOver(self, x, y):
        return self.x < x < self.x + self.w and self.y < y < self.h + self.y

cap = cv2.VideoCapture(0)
tracker = HandTracker(detectionCon=0.8)
keyboard = Controller()

w, h = 80, 60
startX, startY = 40, 200
is_special = False

numbers = list("1234567890")
letters = list("QWERTYUIOPASDFGHJKLZXCVBNM")
special_chars = list("!@#$%^&*()_+-=[]{}|;:',.<>?/\\")
control_keys = ["Space", "clr", "<--", "Enter", "Switch"]

keys = []

def generate_keys():
    global keys
    keys = []
    for i, n in enumerate(numbers):
        keys.append(Key(startX + i * w + i * 5, startY - h - 5, w, h, n))
    
    current_keys = letters if not is_special else special_chars
    for i, l in enumerate(current_keys):
        if i < 10:
            keys.append(Key(startX + i * w + i * 5, startY, w, h, l))
        elif i < 19:
            keys.append(Key(startX + (i - 10) * w + (i - 10) * 5, startY + h + 5, w, h, l))
        else:
            keys.append(Key(startX + (i - 19) * w + (i - 19) * 5, startY + 2 * h + 10, w, h, l))
    
    keys.append(Key(startX, startY + 3 * h + 15, 2 * w, h, "Switch"))
    keys.append(Key(startX + 2 * w + 10, startY + 3 * h + 15, 3 * w, h, "Space"))
    keys.append(Key(startX + 5 * w + 30, startY + 3 * h + 15, 2 * w, h, "<--"))
    keys.append(Key(startX + 7 * w + 50, startY + 3 * h + 15, 2 * w, h, "clr"))
    keys.append(Key(startX + 9 * w + 70, startY + 3 * h + 15, 2 * w, h, "Enter"))

generate_keys()
textBox = Key(startX, startY - 2 * h - 10, 10 * w + 9 * 5, h, '')

ptime = 0
click_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))
    frame = cv2.flip(frame, 1)
    frame = tracker.findHands(frame)
    lmList = tracker.getPosition(frame, draw=False)

    if lmList and tracker.isLeftHand():
        indexTip = lmList[8][1:3]
        thumbTip = lmList[4][1:3]

        for key in keys:
            if key.isOver(*indexTip):
                key.drawKey(frame, alpha=0.1)
            else:
                key.drawKey(frame, alpha=0.5)

            if key.isOver(*indexTip) and key.isOver(*thumbTip):
                current_time = time.time()
                if current_time - click_time > 1:
                    if key.text == "Space":
                        textBox.text += " "
                        keyboard.press(PynputKey.space)
                        keyboard.release(PynputKey.space)
                    elif key.text == "<--":
                        textBox.text = textBox.text[:-1]
                        keyboard.press(PynputKey.backspace)
                        keyboard.release(PynputKey.backspace)
                    elif key.text == "clr":
                        textBox.text = ""
                    elif key.text == "Enter":
                        textBox.text += "\n"
                        keyboard.press(PynputKey.enter)
                        keyboard.release(PynputKey.enter)
                    elif key.text == "Switch":
                        is_special = not is_special
                        generate_keys()
                    else:
                        textBox.text += key.text
                        keyboard.press(key.text)
                        keyboard.release(key.text)
                    click_time = current_time

    textBox.drawKey(frame, (255, 255, 255), (0, 0, 0), alpha=0.3)
    ctime = time.time()
    fps = int(1 / (ctime - ptime))
    ptime = ctime
    cv2.putText(frame, f"{fps} FPS", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow('Virtual Keyboard', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
