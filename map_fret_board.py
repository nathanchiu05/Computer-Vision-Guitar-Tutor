import cv2
import cv2.aruco as aruco
import numpy as np

cap = cv2.VideoCapture(0)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000) #dictionary conaining the aruco markers
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

valid_ids = {0,1,2,3}


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

                # Draw marker outline on mirrored frame
                aruco.drawDetectedMarkers(display, [flipped_corners], ids[i])

                # Draw readable text at top-left corner
                c = flipped_corners[0]
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

    # If we have all 4 corners, draw rectangle
    if len(quad_points) == 4:
        pts = np.array([quad_points["TL"], quad_points["TR"],
                        quad_points["BR"], quad_points["BL"]], dtype=np.int32)
        cv2.polylines(display, [pts], isClosed=True, color=(0,0,255), thickness=3)

    cv2.imshow("ArUco Detection", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


