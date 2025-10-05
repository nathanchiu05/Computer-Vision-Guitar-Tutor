import cv2
import mediapipe as mp
from map_hands import get_fingertip_positions
from map_fret_board import map_guitar

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# remove thumb connections from skeleton
thumb_indices = {(0,1), (1,2), (2,3), (3,4)}
custom_connections = [
    conn for conn in mp_hands.HAND_CONNECTIONS if conn not in thumb_indices
]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    zoom_factor = 1.00   #zoomfacto
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    x1, y1 = (w - new_w) // 2, (h - new_h) // 2
    x2, y2 = x1 + new_w, y1 + new_h
    cropped = frame[y1:y2, x1:x2]
    frame = cv2.resize(cropped, (w, h))

    #keep a copy of the raw frame
    raw_frame = frame.copy()

    # run guitar mapping for visuals
    display = map_guitar(frame)

    # run hand mapping on raw frame (NOT display)
    _, fingertips, landmarks_list = get_fingertip_positions(raw_frame)

    # Draw fingertip dots (flat dict now)
    for name, (x, y) in fingertips.items():
        cv2.circle(display, (x, y), 8, (0, 255, 0), -1)

    cv2.imshow("Hand + Guitar Tracking", display)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()