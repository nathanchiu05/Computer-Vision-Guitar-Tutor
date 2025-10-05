import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands 
hands = mp_hands.Hands(
    static_image_mode=False, #video feed
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

FINGER_TIPS = [8, 12, 16, 20]  #index, middle, ring, pinky

def get_fingertip_positions(frame):
    frame = cv2.flip(frame, 1) 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    tips = []
    landmarks_list = []

    if results.multi_hand_landmarks: #if hands are detected
        h, w, _ = frame.shape #height, width, channels (shape gives dimensions of image)
        
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks_list.append(hand_landmarks)  # keep for drawing
            
            for finger_id in FINGER_TIPS: #parse through fingertip ids and get coords
                x = int(hand_landmarks.landmark[finger_id].x * w)
                y = int(hand_landmarks.landmark[finger_id].y * h)
                tips.append((x, y))

    return frame, tips, landmarks_list #return flipped frame, fingertip coords, and landmarks for drawing skeleton