import cv2
import mediapipe as mp
from collections import deque
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,          # only track one hand
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}

# History buffer for smoothing (5 frames of memory)
finger_history = {name: deque(maxlen=5) for name in FINGER_TIPS.keys()}

def get_fingertip_positions(frame):
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    tips = {}
    landmarks_list = []

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks_list.append(hand_landmarks)

        for name, finger_id in FINGER_TIPS.items():
            lm = hand_landmarks.landmark[finger_id]
            x = int(lm.x * w)
            y = int(lm.y * h)

            # Always append (no visibility filter)
            finger_history[name].append((x, y))

            # Average over history
            avg_x = int(np.mean([p[0] for p in finger_history[name]]))
            avg_y = int(np.mean([p[1] for p in finger_history[name]]))
            tips[name] = (avg_x, avg_y)

    return frame, tips, landmarks_list