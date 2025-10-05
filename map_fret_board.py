import cv2
import cv2.aruco as aruco
import numpy as np
from collections import deque

cap = cv2.VideoCapture(0)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000) #dictionary conaining the aruco markers
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

valid_ids = {0,1,2,3}

history = {0: deque(maxlen=5), 1: deque(maxlen=5), 2: deque(maxlen=5), 3: deque(maxlen=5)}
last_seen = {}

# Standard tuning (low E to high E)
string_labels = ["E", "A", "D", "G", "B", "E"]

while True:
    ret, frame = cap.read() #capture frame-by-frame
    if not ret:
        break

    h, w = frame.shape[:2] #Get frame dimensions

    #Run detection on the non-flipped frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert to grayscale for easier detection
    corners, ids, rejected = detector.detectMarkers(gray)

    #Flip the frame for display (mirror view)
    display = cv2.flip(frame, 1)

    quad_points={}

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in valid_ids:
                #Flip marker corners horizontally for display coords
                flipped_corners = corners[i].copy()
                flipped_corners[0,:,0] = w - corners[i][0,:,0]

                history[marker_id].append(flipped_corners[0])
                avg_c = np.mean(history[marker_id], axis=0)

                # Save smoothed as last seen
                last_seen[marker_id] = avg_c
                c = avg_c

                # Draw marker outline on mirrored frame
                aruco.drawDetectedMarkers(display, [flipped_corners], ids[i])

                # Draw readable text at top-left corner
                # c = flipped_corners[0]
                top_left = tuple(c[0].astype(int))
                cv2.putText(display, f"ID:{marker_id}", top_left,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                if marker_id == 0:  # top-left
                    quad_points["TL"] = c[1]  # TL corner, 0 = top-left
                elif marker_id == 1:  # top-right
                    quad_points["TR"] = c[0]  # TR corner
                elif marker_id == 2:  # bottom-right
                    quad_points["BR"] = c[3]  # BR corner
                elif marker_id == 3:  # bottom-left
                    quad_points["BL"] = c[2]  # BL corner

        # If some IDs are missing this frame, reuse last seen smoothed values
        for mid in valid_ids:
            if mid not in quad_points and mid in last_seen:
                c = last_seen[mid]
                if mid == 0:   quad_points["TL"] = c[1]
                elif mid == 1: quad_points["TR"] = c[0]
                elif mid == 2: quad_points["BR"] = c[3]
                elif mid == 3: quad_points["BL"] = c[2]

    # If we have all 4 corners, draw rectangle and strings
    if len(quad_points) == 4:
        pts = np.array([quad_points["TL"], quad_points["TR"],
                        quad_points["BR"], quad_points["BL"]], dtype=np.int32)
        cv2.polylines(display, [pts], isClosed=True, color=(0,0,255), thickness=3)

        # Interpolate strings between left and right edges
        left_edge  = np.linspace(quad_points["BR"], quad_points["BL"], 6)
        right_edge = np.linspace(quad_points["TR"], quad_points["TL"], 6)

        for i in range(6):
            p1 = tuple(left_edge[i].astype(int))
            p2 = tuple(right_edge[i].astype(int))

            cv2.line(display, p1, p2, (155,255,0), 2)

            # Put string label on left side
            label_pos = (p2[0] + 10, p2[1] + 5)  # move right of the string line
            cv2.putText(display, string_labels[i], label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("ArUco Fretboard with Strings", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


