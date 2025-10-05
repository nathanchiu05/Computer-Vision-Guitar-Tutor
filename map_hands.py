import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Remove thumb connections (0-1, 1-2, 2-3, 3-4)
thumb_indices = {(0,1), (1,2), (2,3), (3,4)}
custom_connections = [
    conn for conn in mp_hands.HAND_CONNECTIONS if conn not in thumb_indices
]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape

            # Draw only fingertips for index–pinky (no thumb)
            for fid in [8, 12, 16, 20]:
                x = int(hand_landmarks.landmark[fid].x * w)
                y = int(hand_landmarks.landmark[fid].y * h)
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # Draw skeleton without thumb
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                custom_connections
            )

    cv2.imshow("Test Hands", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()