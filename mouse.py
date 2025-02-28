import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

# Initialize timers
last_left_click_time = 0
last_right_click_time = 0
last_switch_time = 0
last_search_time = 0
last_minimize_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            
            if handedness == "Right":  # Process only the right hand
                lm = hand_landmarks.landmark
                
                # Get finger tip positions
                index_finger = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_finger = lm[mp_hands.HandLandmark.RING_FINGER_TIP]
                thumb = lm[mp_hands.HandLandmark.THUMB_TIP]
                pinky = lm[mp_hands.HandLandmark.PINKY_TIP]
                
                # Convert to screen coordinates
                index_x, index_y = int(index_finger.x * w), int(index_finger.y * h)
                middle_x, middle_y = int(middle_finger.x * w), int(middle_finger.y * h)
                
                # Check which fingers are up
                fingers = [
                    index_finger.y < lm[mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
                    middle_finger.y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,
                    ring_finger.y < lm[mp_hands.HandLandmark.RING_FINGER_DIP].y,
                    thumb.y < lm[mp_hands.HandLandmark.THUMB_IP].y,
                    pinky.y < lm[mp_hands.HandLandmark.PINKY_PIP].y  # Adjusted pinky detection
                ]
                
                current_time = time.time()

                # **Check if any gesture is active (prevents cursor movement)**
                performing_gesture = False

                # Left Click: Index & Middle Finger Together
                if fingers[0] and fingers[1] and abs(index_x - middle_x) < 20:
                    if current_time - last_left_click_time > 2:
                        pyautogui.click()
                        last_left_click_time = current_time
                        performing_gesture = True
                
                # Right Click: Index & Middle Finger in V Shape
                if fingers[0] and fingers[1] and abs(index_x - middle_x) > 40:
                    if current_time - last_right_click_time > 2:
                        pyautogui.rightClick()
                        last_right_click_time = current_time
                        performing_gesture = True

                # Screenshot: Index, Middle, and Ring Fingers Up
                if fingers[0] and fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                    pyautogui.screenshot("screenshot.png")
                    performing_gesture = True
                
                # Switch Apps (Alt+Tab): Only Thumb Up
                if fingers[3] and not any(fingers[:3]) and not fingers[4]:
                    if current_time - last_switch_time > 2:
                        pyautogui.hotkey("alt", "tab")
                        last_switch_time = current_time
                        performing_gesture = True
                
                # Windows Search: All Fingers Up
                if all(fingers):
                    if current_time - last_search_time > 4:
                        pyautogui.press("win")
                        last_search_time = current_time
                        performing_gesture = True
                
                # Minimize All Windows (Win+M): Only Pinky Up
                if fingers[4] and not any(fingers[:4]):
                    if current_time - last_minimize_time > 4:
                        pyautogui.hotkey("win", "m")
                        last_minimize_time = current_time
                        performing_gesture = True
                
                # **Cursor Movement (Only if no other gesture is running)**
                if fingers[0] and not performing_gesture:
                    pyautogui.moveTo(index_finger.x * screen_width, index_finger.y * screen_height)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Hand Gesture Control", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
