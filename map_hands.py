import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands 
hands = mp_hands.Hands(
    static_image_mode=False,  # video feed
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}

def get_fingertip_positions(frame):
    frame = cv2.flip(frame, 1) 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    tips = {"Left": {}, "Right": {}}  # store fingers per hand
    landmarks_list = []

    if results.multi_hand_landmarks and results.multi_handedness:
        h, w, _ = frame.shape
        
        # pair hand landmarks with handedness
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            landmarks_list.append(hand_landmarks)

            for name, finger_id in FINGER_TIPS.items():
                x = int(hand_landmarks.landmark[finger_id].x * w)
                y = int(hand_landmarks.landmark[finger_id].y * h)
                tips[label][name] = (x, y)   # ✅ store under Left/Right

    return frame, tips, landmarks_list