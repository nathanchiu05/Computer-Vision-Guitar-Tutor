import cv2
import mediapipe as mp
import math
from map_hands import get_fingertip_positions
from map_fret_board import map_guitar
from graphics_code import draw_chord_diagram
import graphics_code

from match_chord import match_chord

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

thumb_indices = {(0,1), (1,2), (2,3), (3,4)}
custom_connections = [conn for conn in mp_hands.HAND_CONNECTIONS if conn not in thumb_indices]

string_labels = ["E", "A", "D", "G", "B", "E"]

cap = cv2.VideoCapture(0)

current_chord = "C"  # <-- set this dynamically if needed


# Utility: compute accuracy of observed fingertips vs expected chord points
def compute_chord_accuracy(expected_positions, observed_points, max_distance=80):
    """
    expected_positions: dict mapping (s_idx, fret) -> (x_dst, y_dst)
    observed_points: list of (bx, by) observed in same dst coords
    max_distance: distance considered full-miss (>= max_distance -> 0 score for that string)

    Returns: (percent_score, details)
    details: list of (s_idx, expected_fret, matched_bool, distance)
    """
    # Build per-string expected (pick fretted positions only)
    per_string_expected = {}
    for (s_idx, fret), pos in expected_positions.items():
        if fret is not None and fret > 0:
            # prefer higher fret if duplicates (shouldn't happen normally)
            per_string_expected[s_idx] = (fret, pos)

    details = []
    correct = 0
    total = 0

    for s_idx, (fret_needed, pos) in per_string_expected.items():
        total += 1
        best_d = float('inf')
        for (bx, by) in observed_points:
            d = math.hypot(bx - pos[0], by - pos[1])
            if d < best_d:
                best_d = d

        matched = best_d <= max_distance
        if matched:
            correct += 1
        details.append((s_idx, fret_needed, matched, best_d if best_d != float('inf') else None))

    pct = int(100 * correct / total) if total else 0
    return pct, details


def compute_accuracy_from_lists(expected_list, observed_list, max_distance=80):
    """Compute percent of expected points that have an observed point within max_distance."""
    correct = 0
    total = len(expected_list)
    details = []
    for ex in expected_list:
        best = float('inf')
        for ob in observed_list:
            d = math.hypot(ex[0]-ob[0], ex[1]-ob[1])
            if d < best:
                best = d
        matched = best <= max_distance
        if matched:
            correct += 1
        details.append((ex, best, matched))
    pct = int(100 * correct / total) if total else 0
    return pct, details


while True:
    ret, frame = cap.read()
    if not ret:
        break

    raw_frame = frame.copy()
    display, fret_positions, string_positions = map_guitar(frame)
    _, fingertips, landmarks_list = get_fingertip_positions(raw_frame)

    # After mapping the guitar and getting fret/string positions:
    if fret_positions and string_positions:
        fret_xs = [x for (x,y) in fret_positions]
        string_ys = [y for (x,y) in string_positions]

        def nearest(value, candidates, threshold=50):
            """Find nearest candidate within threshold, return None if too far"""
            if not candidates:
                return None
            
            # Find the closest candidate
            closest_idx = min(range(len(candidates)), key=lambda i: abs(candidates[i] - value))
            closest_distance = abs(candidates[closest_idx] - value)
            
            # Only return the index if within threshold, otherwise return None
            if closest_distance <= threshold:
                return closest_idx
            else:
                return None

        # Draw fingertip positions
        for name, (x, y) in fingertips.items():
            fret_idx = nearest(x, fret_xs)
            string_idx = nearest(y, string_ys)

            if fret_idx is not None and string_idx is not None:
                fret_idx += 1
                note_text = f"{name}: String {string_labels[string_idx]}, Fret {fret_idx}"
                print(note_text)
                # cv2.putText(display, note_text, (x+10, y-10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            else:
                # Finger is not near any string - show as open/not playing
                note_text = f"{name}: Not on string"
                print(note_text)
                # cv2.putText(display, note_text, (x+10, y-10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 2)

            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)

    # ===============================================================
    # DRAW CURRENT CHORD ON FRETBOARD (yellow overlay)
    # ===============================================================
    # Prepare lists for accuracy checking (always defined)
    expected_screen_positions = []
    observed_screen_points = [ (x,y) for (_, (x,y)) in fingertips.items() ]

    if current_chord and fret_positions and string_positions:
        for string_idx, fret in enumerate(graphics_code.CHORD_LIBRARY[current_chord]['frets']):
            finger_num = graphics_code.CHORD_LIBRARY[current_chord]['fingers'][string_idx]
            if fret is not None and fret > 0 and finger_num:
                # Map string index to y position
                y = string_ys[::-1][string_idx]
                # Map fret index to x position
                if fret <= len(fret_xs):
                    x = fret_xs[fret-1]
                else:
                    x = fret_xs[-1]
                
                # Draw yellow filled circle + white outline
                cv2.circle(display, (x, y), 8, (0, 255, 255), -1)
                cv2.circle(display, (x, y), 8, (255, 255, 255), 2)
                cv2.putText(display, str(finger_num), (x-7, y+7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                # record expected screen-space position for accuracy check
                expected_screen_positions.append((x,y))

    # compute accuracy between observed_screen_points and expected_screen_positions (only when expected exists)
    if expected_screen_positions:
        pct, details = compute_accuracy_from_lists(expected_screen_positions, observed_screen_points, max_distance=60)
        cv2.putText(display, f"Accuracy: {pct}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if pct==100 else (0,165,255), 2)
    else:
        # no expected points (open chord/no fretted notes) - show N/A
        cv2.putText(display, f"Accuracy: N/A", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)


    # Draw chord diagram on the frame
    draw_chord_diagram(
        display,
        x=20,
        y=display.shape[0]-220,
        width=200,
        height=180,
        show_instructions=True,
        current_chord=current_chord
    )

    # Display current chord
    cv2.putText(display, f"Current Chord: {current_chord}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display instructions
    cv2.putText(display, "Press 1-8 to change chords, ESC to exit", (20, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Hand + Guitar Tracking", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('1'):
        current_chord = "A"
    elif key == ord('2'):
        current_chord = "Am"
    elif key == ord('3'):
        current_chord = "C"
    elif key == ord('4'):
        current_chord = "D"
    elif key == ord('5'):
        current_chord = "Dm"
    elif key == ord('6'):
        current_chord = "E"
    elif key == ord('7'):
        current_chord = "Em"
    elif key == ord('8'):
        current_chord = "G"


cap.release()
cv2.destroyAllWindows()
