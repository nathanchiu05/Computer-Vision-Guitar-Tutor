from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import time

# Import your chord matching
from match_chord import match_chord

app = Flask(__name__)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Constants
NUM_STRINGS = 6
NUM_FRETS = 12
STRING_NAMES = ["E", "B", "G", "D", "A", "E"]
FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky"]

CHORD_LIBRARY = {
    "C": {"frets": [None, 3, 2, 0, 1, 0], "fingers": [None, 3, 2, None, 1, None], "name": "C Major"},
    "D": {"frets": [None, None, 0, 2, 3, 2], "fingers": [None, None, None, 1, 3, 2], "name": "D Major"},
    "E": {"frets": [0, 2, 2, 1, 0, 0], "fingers": [None, 2, 3, 1, None, None], "name": "E Major"},
    "G": {"frets": [3, 2, 0, 0, 0, 3], "fingers": [3, 2, None, None, None, 4], "name": "G Major"},
    "A": {"frets": [None, 0, 2, 2, 2, 0], "fingers": [None, None, 1, 2, 3, None], "name": "A Major"},
    "Em": {"frets": [0, 2, 2, 0, 0, 0], "fingers": [None, 2, 3, None, None, None], "name": "E Minor"},
    "Am": {"frets": [None, 0, 2, 2, 1, 0], "fingers": [None, None, 2, 3, 1, None], "name": "A Minor"},
    "Dm": {"frets": [None, None, 0, 2, 3, 1], "fingers": [None, None, None, 2, 4, 1], "name": "D Minor"},
}

@dataclass
class FretboardRegion:
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    fret_markers: List[Tuple[int, int, int, int]]
    quad_corners: Optional[np.ndarray] = None

@dataclass
class FingerPosition:
    finger_name: str
    string_num: int
    fret_num: int

class FingerTracker:
    def __init__(self):
        self.current_positions: List[FingerPosition] = []
    
    def detect_position(self, tip_x, tip_y, region: FretboardRegion) -> Optional[Tuple[int, int]]:
        if not region:
            return None
        
        if hasattr(region, 'quad_corners') and region.quad_corners is not None:
            corners = region.quad_corners
            result = cv2.pointPolygonTest(corners.reshape(-1, 2).astype(np.int32), (tip_x, tip_y), False)
            if result < 0:
                return None
            
            left_top = corners[0, 0]
            left_bottom = corners[3, 0]
            right_top = corners[1, 0]
            right_bottom = corners[2, 0]
            
            def point_to_line_t(p, line_start, line_end):
                line_vec = line_end - line_start
                point_vec = p - line_start
                line_len_sq = np.dot(line_vec, line_vec)
                if line_len_sq == 0:
                    return 0
                t = np.dot(point_vec, line_vec) / line_len_sq
                return max(0, min(1, t))
            
            point_arr = np.array([tip_x, tip_y])
            t_left = point_to_line_t(point_arr, left_top, left_bottom)
            t_right = point_to_line_t(point_arr, right_top, right_bottom)
            t_string = (t_left + t_right) / 2
            string_idx = int(t_string * NUM_STRINGS)
            string_idx = max(0, min(NUM_STRINGS - 1, string_idx))
            
            if region.fret_markers:
                min_x = min(corners[:, 0, 0])
                fret_x_positions = [min_x] + [m[0] for m in region.fret_markers]
                for i in range(len(fret_x_positions) - 1):
                    if fret_x_positions[i] <= tip_x <= fret_x_positions[i + 1]:
                        return string_idx + 1, i + 1
                if tip_x > fret_x_positions[-1]:
                    return string_idx + 1, len(fret_x_positions)
        
        return None
    
    def update_positions(self, hand_landmarks, image_width, image_height, region: Optional[FretboardRegion]) -> List[Tuple[int, int, FingerPosition]]:
        if not region:
            return []
        
        self.current_positions.clear()
        positions = []
        
        finger_tips = [
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP
        ]
        
        finger_mcps = [
            mp_hands.HandLandmark.INDEX_FINGER_MCP,
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            mp_hands.HandLandmark.RING_FINGER_MCP,
            mp_hands.HandLandmark.PINKY_MCP
        ]
        
        for i, (tip_idx, mcp_idx) in enumerate(zip(finger_tips, finger_mcps)):
            tip = hand_landmarks.landmark[tip_idx]
            mcp = hand_landmarks.landmark[mcp_idx]
            tip_x = int(tip.x * image_width)
            tip_y = int(tip.y * image_height)
            
            if tip.z < mcp.z - 0.02:
                result = self.detect_position(tip_x, tip_y, region)
                if result:
                    string_num, fret_num = result
                    position = FingerPosition(
                        finger_name=FINGER_NAMES[i],
                        string_num=string_num,
                        fret_num=fret_num
                    )
                    self.current_positions.append(position)
                    positions.append((tip_x, tip_y, position))
        
        return positions

def draw_fretboard(image, region: FretboardRegion, chord=None):
    if hasattr(region, 'quad_corners') and region.quad_corners is not None:
        corners = region.quad_corners.astype(np.int32)
        cv2.polylines(image, [corners], True, (0, 255, 0), 2)
        
        # Draw strings
        for i in range(NUM_STRINGS):
            t = (i + 0.5) / NUM_STRINGS
            left_point = corners[0, 0] + t * (corners[3, 0] - corners[0, 0])
            right_point = corners[1, 0] + t * (corners[2, 0] - corners[1, 0])
            cv2.line(image, 
                    (int(left_point[0]), int(left_point[1])),
                    (int(right_point[0]), int(right_point[1])),
                    (0, 200, 0), 1)
            
            label_x = int(left_point[0] - 25)
            label_y = int(left_point[1] + 5)
            cv2.putText(image, STRING_NAMES[i], (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw frets
        for fret in range(1, NUM_FRETS):
            t = fret / NUM_FRETS
            top = corners[0, 0] + t * (corners[1, 0] - corners[0, 0])
            bottom = corners[3, 0] + t * (corners[2, 0] - corners[3, 0])
            cv2.line(image, 
                     (int(top[0]), int(top[1])),
                     (int(bottom[0]), int(bottom[1])),
                     (0, 200, 0), 2)
            cv2.putText(image, str(fret), (int(top[0]) - 8, int(top[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Nut (thick left line)
        cv2.line(image, 
                 (int(corners[0, 0][0]), int(corners[0, 0][1])),
                 (int(corners[3, 0][0]), int(corners[3, 0][1])),
                 (0, 255, 0), 4)

# Global state
tracker = FingerTracker()
manual_region = None

def generate_frames():
    global manual_region
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Hardcode fretboard region for now (you'll adjust these coordinates)
    # Format: top-left corner, top-right, bottom-right, bottom-left
    manual_region = FretboardRegion(
        top_left=(200, 200),
        bottom_right=(1000, 400),
        fret_markers=[]
    )
    
    # Create quad corners
    tl_x, tl_y = manual_region.top_left
    br_x, br_y = manual_region.bottom_right
    manual_region.quad_corners = np.float32([
        [tl_x, tl_y],
        [br_x, tl_y],
        [br_x, br_y],
        [tl_x, br_y]
    ]).reshape(-1, 1, 2)
    
    # Create fret markers
    fret_width = (br_x - tl_x) / NUM_FRETS
    manual_region.fret_markers = [(int(tl_x + i * fret_width), tl_y, 2, br_y - tl_y)
                                   for i in range(1, NUM_FRETS + 1)]
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw fretboard
        if manual_region:
            draw_fretboard(frame, manual_region)
        
        # Draw hands and track fingers
        matched_chord = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                positions = tracker.update_positions(hand_landmarks, w, h, manual_region)
                
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                )
                
                # Draw purple fingertips
                fingertip_ids = [
                    mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.PINKY_TIP
                ]
                for tip_id in fingertip_ids:
                    tip = hand_landmarks.landmark[tip_id]
                    cx, cy = int(tip.x * w), int(tip.y * h)
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 255), -1)
                
                # Match chord
                if tracker.current_positions:
                    finger_positions = [(FINGER_NAMES.index(pos.finger_name) + 1, pos.string_num, pos.fret_num) 
                                       for pos in tracker.current_positions]
                    matched_chord = match_chord(finger_positions)
        
        # Display matched chord
        if matched_chord:
            cv2.putText(frame, f"Chord: {matched_chord}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No chord detected", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Display finger positions
        for i, pos in enumerate(tracker.current_positions[:4]):
            text = f"{pos.finger_name}: S{pos.string_num} F{pos.fret_num}"
            cv2.putText(frame, text, (10, 100 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)