import cv2
import cv2.aruco as aruco
import numpy as np
from collections import deque

# ArUco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

valid_ids = {0,1,2,3}
history = {i: deque(maxlen=5) for i in valid_ids}
last_seen = {}

# Standard tuning (low E to high E)
string_labels = ["E", "A", "D", "G", "B", "E"]

def map_guitar(frame):
    """Process a frame, detect ArUco fretboard, draw frets + strings, return annotated display."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    display = cv2.flip(frame, 1)
    quad_points = {}

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in valid_ids:
                flipped_corners = corners[i].copy()
                flipped_corners[0,:,0] = w - corners[i][0,:,0]

                history[marker_id].append(flipped_corners[0])
                avg_c = np.mean(history[marker_id], axis=0)
                last_seen[marker_id] = avg_c
                c = avg_c

                aruco.drawDetectedMarkers(display, [flipped_corners], ids[i])
                top_left = tuple(c[0].astype(int))
                cv2.putText(display, f"ID:{marker_id}", top_left,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                if marker_id == 0:  quad_points["TL"] = c[1]
                elif marker_id == 1: quad_points["TR"] = c[0]
                elif marker_id == 2: quad_points["BR"] = c[3]
                elif marker_id == 3: quad_points["BL"] = c[2]

        # reuse last seen markers if missing
        for mid in valid_ids:
            if mid not in quad_points and mid in last_seen:
                c = last_seen[mid]
                if mid == 0:   quad_points["TL"] = c[1]
                elif mid == 1: quad_points["TR"] = c[0]
                elif mid == 2: quad_points["BR"] = c[3]
                elif mid == 3: quad_points["BL"] = c[2]

    # draw fretboard if complete
    if len(quad_points) == 4:
        pts = np.array([quad_points["TL"], quad_points["TR"],
                        quad_points["BR"], quad_points["BL"]], dtype=np.int32)
        cv2.polylines(display, [pts], True, (0,0,255), 3)

        TL = np.array(quad_points["TL"], dtype=np.float32)
        TR = np.array(quad_points["TR"], dtype=np.float32)
        BL = np.array(quad_points["BL"], dtype=np.float32)
        BR = np.array(quad_points["BR"], dtype=np.float32)

        left_edge_vec  = BL - TL
        right_edge_vec = BR - TR

        prev_frac = 0
        num_frets = 12
        for n in range(1, num_frets + 1):
            fret_frac = prev_frac + (1 - prev_frac) / 17.817
            fret_left  = TL + left_edge_vec * fret_frac * 2
            fret_right = TR + right_edge_vec * fret_frac * 2
            cv2.line(display, tuple(fret_left.astype(int)), tuple(fret_right.astype(int)), (0, 255, 255), 2)
            cv2.putText(display, f"{n}", tuple((fret_left + [5, -5]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            prev_frac = fret_frac

        upper_edge  = np.linspace(quad_points["BR"], quad_points["BL"], 6)
        lower_edge  = np.linspace(quad_points["TR"], quad_points["TL"], 6)

        for i in range(6):
            p1 = tuple(upper_edge[i].astype(int))
            p2 = tuple(lower_edge[i].astype(int))
            cv2.line(display, p1, p2, (155,255,0), 2)
            label_pos = (p2[0] + 10, p2[1] + 5)
            # cv2.putText(display, string_labels[i], label_pos,
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    return display