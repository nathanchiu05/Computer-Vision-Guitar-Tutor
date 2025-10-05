import cv2
import mediapipe as mp
from map_hands import get_fingertip_positions

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Remove thumb connections (0-1, 1-2, 2-3, 3-4)
thumb_indices = {(0,1), (1,2), (2,3), (3,4)}
custom_connections = [
    conn for conn in mp_hands.HAND_CONNECTIONS if conn not in thumb_indices
]

cap = cv2.VideoCapture(0) # webcam input

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get flipped frame, fingertip coords, and landmarks
    frame, fingertips, landmarks_list = get_fingertip_positions(frame)

    # Debug print
    print(fingertips)

    # Draw skeleton without thumb
    for hand_landmarks in landmarks_list:
        mp_draw.draw_landmarks(frame, hand_landmarks, custom_connections)

    # Draw fingertip dots with Left/Right labels
    for hand_label, fingers in fingertips.items():
        for name, (x, y) in fingers.items():
            # color left vs right differently
            color = (0, 255, 0) if hand_label == "Left" else (0, 0, 255)
            cv2.circle(frame, (x, y), 8, color, -1)
            # cv2.putText(frame, f"{hand_label}-{name}", (x+10, y-10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # esc to exit
        break

cap.release()
cv2.destroyAllWindows()