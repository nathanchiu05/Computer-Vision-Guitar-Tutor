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

# standard tuning
string_labels = ["E", "A", "D", "G", "B", "E"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # keep a copy of the raw frame
    raw_frame = frame.copy()

    # run guitar mapping for visuals
    display, fret_positions, string_positions = map_guitar(frame)

    # run hand mapping on raw frame (NOT display)
    _, fingertips, landmarks_list = get_fingertip_positions(raw_frame)

    # only proceed if guitar frets/strings detected
    if fret_positions and string_positions:
        fret_xs = [x for (x,y) in fret_positions]
        string_ys = [y for (x,y) in string_positions]

        def nearest(value, candidates):
            if not candidates:
                return None
            return min(range(len(candidates)), key=lambda i: abs(candidates[i] - value))

        # Draw fingertip dots (flat dict now)
        for name, (x, y) in fingertips.items():
            fret_idx = nearest(x, fret_xs)
            string_idx = nearest(y, string_ys)

            if fret_idx is not None and string_idx is not None:
                fret_idx += 1  # +1 because frets start at 1
                note_text = f"{name}: String {string_labels[string_idx]}, Fret {fret_idx}"
                print(note_text)  # console output
                cv2.putText(display, note_text, (x+10, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            cv2.circle(display, (x, y), 8, (0, 255, 0), -1)

    cv2.imshow("Hand + Guitar Tracking", display)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()